// Package scheduler runs the kernel's tick loop and owns the per-agent
// transport bindings. One tick proceeds in six phases per the Step 2
// brief, with one small naming divergence documented in PAPER_NOTES:
//
//   0. handler.DrainQueueAt(t)        — apply Retune / Close / Kill
//                                      mutations queued *for* tick t
//                                      (so step 4 snapshots them)
//   1. dispatch views                 — BuildView → transport.Tick
//   2/3. collect + commit             — (synchronous, fused with step 1
//                                      under the in-process transport);
//                                      commit order is
//                                      (emitter_pubkey lex asc, then
//                                      stable arrival index)
//   4. snapshot L_cog_ind             — one CogIndSnapshot per live
//                                      agent, kernel-signed
//   5. scenario ops                   — drain scenarioOps[t]; ops are
//                                      handler-method closures, may
//                                      queue further mutations for t+1
//   6. tick++
//
// Brief deviation: the brief's step 5 reads "execute ops scheduled for
// tick t+1". We instead drain ops keyed by the *execution* tick — i.e.,
// `ScheduleAt(N, op)` runs `op` at step 5 of tick N. The user-facing
// "Retune at tick 5" idiom then matches: Retune called during step 5 of
// tick 5 has effective_tick = 6, so tick 6's L_cog_ind snapshot sees
// the new theta (matching experiment 006's expectation). PAPER_NOTES
// records this convention.
package scheduler

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/elicit"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/schema"
	"github.com/byzminds/byzminds/kernel/internal/view"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	kernelv1 "github.com/byzminds/byzminds/proto/kernelv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

// CommitNotifier is the optional second method an AgentTransport may
// implement to receive the kernel's commit receipt after the scheduler
// finishes its sorted-commit phase. The gRPC subscriber bridge
// implements this so SubmitEvent's awaiting goroutine can return a
// receipt to the agent. In-process stubs do not implement it.
type CommitNotifier interface {
	NotifyCommit(env *eventsv1.EventEnvelope, receipt *kernelv1.CommitReceipt)
}

// DefaultTickTimeout matches the brief's "30s budget" for agent
// emission. Step 3 will revisit once LLM latency is known.
const DefaultTickTimeout = 30 * time.Second

// AgentTransport is the per-agent send/receive shim. The in-process
// stub agents implement it as a synchronous closure; the gRPC bridge
// will implement it as a channel pair (Step 2 plumbs, Step 3+ exercises).
type AgentTransport interface {
	// Tick delivers the view to the agent and returns the agent's
	// emitted envelope, or an error (treated as a tick-timeout).
	Tick(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error)
}

// AgentTransportFunc adapts a plain closure.
type AgentTransportFunc func(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error)

// Tick implements AgentTransport.
func (f AgentTransportFunc) Tick(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	return f(ctx, v)
}

// Scheduler is the tick-loop owner.
type Scheduler struct {
	ls *ledger.LedgerSet
	h  *handler.Handler

	kernelPub  ed25519.PublicKey
	kernelPriv ed25519.PrivateKey

	tickTimeout time.Duration

	// K_elicit (Step 4): how often phase 3.5 runs. 0 disables, 3 default.
	kElicit uint32

	mu          sync.Mutex
	transports  map[string]AgentTransport
	scenarioOps map[uint64][]func() error

	currentTick atomic.Uint64
	paused      atomic.Bool
	stepBudget  atomic.Int32

	resumeCh chan struct{} // signalled when paused → not-paused, or step granted
}

// New constructs a scheduler bound to ls and h. The handler's
// SchedulerControl is attached automatically. K_elicit defaults to
// elicit.DefaultKElicit (3); change via SetKElicit before RunUntil.
func New(ls *ledger.LedgerSet, h *handler.Handler, tickTimeout time.Duration) *Scheduler {
	if tickTimeout <= 0 {
		tickTimeout = DefaultTickTimeout
	}
	s := &Scheduler{
		ls:          ls,
		h:           h,
		kernelPub:   ls.KernelPriv().Public().(ed25519.PublicKey),
		kernelPriv:  ls.KernelPriv(),
		tickTimeout: tickTimeout,
		kElicit:     elicit.DefaultKElicit,
		transports:  make(map[string]AgentTransport),
		scenarioOps: make(map[uint64][]func() error),
		resumeCh:    make(chan struct{}, 1),
	}
	h.SetScheduler(s)
	return s
}

// AttachAgent binds the agent_id (must already be spawned via Handler)
// to a transport. The scheduler will call transport.Tick once per tick
// from the agent's spawn_tick forward.
func (s *Scheduler) AttachAgent(agentID string, t AgentTransport) error {
	if _, ok := s.h.LookupAgent(agentID); !ok {
		return fmt.Errorf("scheduler: agent %q is not spawned", agentID)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.transports[agentID]; dup {
		return fmt.Errorf("scheduler: agent %q already has a transport", agentID)
	}
	s.transports[agentID] = t
	return nil
}

// SetKElicit overrides the default K_elicit (3). Pass 0 to disable
// elicitation entirely (Step 4 brief's "configurable per scenario"
// option for stubs that don't implement elicit semantics).
func (s *Scheduler) SetKElicit(k uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.kElicit = k
}

// HasTransport reports whether agentID has been attached. Used by
// orchestrators that need to wait for a gRPC bridge (e.g., the Python
// adapter's Subscribe call) to register before advancing the scheduler.
func (s *Scheduler) HasTransport(agentID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.transports[agentID]
	return ok
}

// WaitForAllLiveAgentsAttached blocks until every currently-live agent
// has a transport attached, or ctx is cancelled. Polls at 20ms intervals.
// Used by orchestrators (byzminds-run, test harnesses) to align scheduler
// boot with cross-process agent subscriptions.
func (s *Scheduler) WaitForAllLiveAgentsAttached(ctx context.Context) error {
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		live := s.h.LiveAgents()
		allAttached := true
		s.mu.Lock()
		for _, id := range live {
			if _, ok := s.transports[id]; !ok {
				allAttached = false
				break
			}
		}
		s.mu.Unlock()
		if allAttached {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// ScheduleAt queues an op to execute at step 5 of tick `execTick`. Ops
// run in the order they were scheduled.
func (s *Scheduler) ScheduleAt(execTick uint64, op func() error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scenarioOps[execTick] = append(s.scenarioOps[execTick], op)
}

// CurrentTick is the SchedulerControl read accessor.
func (s *Scheduler) CurrentTick() uint64 { return s.currentTick.Load() }

// Pause halts the loop at the next tick boundary.
func (s *Scheduler) Pause() {
	s.paused.Store(true)
	s.stepBudget.Store(0)
}

// Resume releases the loop to advance freely.
func (s *Scheduler) Resume() {
	s.paused.Store(false)
	select {
	case s.resumeCh <- struct{}{}:
	default:
	}
}

// Step grants `n` more tick boundaries to a paused scheduler.
func (s *Scheduler) Step(n uint32) {
	if n == 0 {
		n = 1
	}
	s.stepBudget.Add(int32(n))
	select {
	case s.resumeCh <- struct{}{}:
	default:
	}
}

// RunUntil advances ticks until currentTick == lastTickExclusive or
// ctx is cancelled. Pause/Step gating is respected.
func (s *Scheduler) RunUntil(ctx context.Context, lastTickExclusive uint64) error {
	for s.currentTick.Load() < lastTickExclusive {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if s.paused.Load() && s.stepBudget.Load() <= 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-s.resumeCh:
			case <-time.After(50 * time.Millisecond):
			}
			continue
		}
		if err := s.runOneTick(ctx); err != nil {
			return err
		}
		if s.paused.Load() {
			s.stepBudget.Add(-1)
		}
	}
	return nil
}

// runOneTick is the six-phase loop body.
func (s *Scheduler) runOneTick(ctx context.Context) error {
	t := s.currentTick.Load()

	// Phase 0: drain deferred mutations queued for tick t.
	if err := s.h.DrainQueueAt(t); err != nil {
		return fmt.Errorf("scheduler: drain handler queue at tick %d: %w", t, err)
	}

	// Phases 1+2 fused (synchronous in-process transport): for each
	// live agent, build its view, call transport.Tick, collect envelope.
	live := s.h.LiveAgents()
	pendings := make([]*pendingEvent, 0, len(live))
	deadline := time.Now().Add(s.tickTimeout)
	for _, agentID := range live {
		s.mu.Lock()
		transport, ok := s.transports[agentID]
		s.mu.Unlock()

		v, err := view.BuildView(s.h, s.ls, agentID, t)
		if err != nil {
			return fmt.Errorf("scheduler: build view for %s tick %d: %w", agentID, t, err)
		}
		view.ResolveSenderIDs(v, s.h, s.ls)

		ag, _ := s.h.LookupAgent(agentID)

		if !ok {
			// No transport → synthesize Yield immediately (no waiting).
			env, err := s.synthesizeYield(ag.Pubkey, t)
			if err != nil {
				return err
			}
			s.recordTickTimeout(ag.ID, t)
			pendings = append(pendings, &pendingEvent{env: env, synthetic: true})
			continue
		}

		tickCtx, cancel := context.WithDeadline(ctx, deadline)
		env, err := transport.Tick(tickCtx, v)
		cancel()
		if err != nil || env == nil {
			synth, serr := s.synthesizeYield(ag.Pubkey, t)
			if serr != nil {
				return serr
			}
			s.recordTickTimeout(ag.ID, t)
			pendings = append(pendings, &pendingEvent{env: synth, synthetic: true, transport: transport})
			continue
		}
		pendings = append(pendings, &pendingEvent{env: env, synthetic: false, transport: transport})
	}

	// Phase 3: deterministic commit order. Primary: emitter_pubkey lex.
	// Secondary: stable arrival index (already preserved by SliceStable).
	sort.SliceStable(pendings, func(i, j int) bool {
		return bytes.Compare(pendings[i].env.GetEmitterPubkey(), pendings[j].env.GetEmitterPubkey()) < 0
	})
	for _, p := range pendings {
		if p.synthetic {
			if _, err := s.ls.Commit(p.env, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
				return fmt.Errorf("scheduler: commit synthetic yield at tick %d: %w", t, err)
			}
			continue
		}
		receipt, err := s.commitAgentEnvelopeWithReceipt(p.env)
		if err != nil {
			return fmt.Errorf("scheduler: commit at tick %d: %w", t, err)
		}
		// Notify the bridge (if this transport is one) so the awaiting
		// SubmitEvent goroutine on the gRPC side can return the receipt
		// to the Python agent.
		if cn, ok := p.transport.(CommitNotifier); ok {
			cn.NotifyCommit(p.env, receipt)
		}
		if !receipt.GetCommitted() {
			// Rejection: receipt already carries the reason; skip the
			// post-commit handler hook for non-committed events.
			continue
		}
		// Post-commit hook for agent-emitted control intentions.
		if err := s.h.HandleAgentControlEvent(p.env, t); err != nil {
			return fmt.Errorf("scheduler: post-commit handler hook: %w", err)
		}
	}

	// Phase 3.5 (Step 4): per-K elicitation pass. For each live agent
	// in sorted agent_id order, if ShouldElicit(t, K), commit an
	// ElicitationRequest to L_ctrl, dispatch an elicit-flavored View
	// to that agent's transport, and (if it returns a DeclareIntent
	// envelope) commit to L_cog_eli. Non-compliant transports (Step 2
	// stubs) yield non-DeclareIntent events which are recorded as
	// MalformedSubmissions on L_ctrl rather than committed.
	//
	// Same-tick ordering: action View first (above), elicit View
	// second. Replay correctness depends on this.
	if elicit.ShouldElicit(t, s.kElicit) {
		if err := s.runElicitPass(ctx, t, pendings); err != nil {
			return fmt.Errorf("scheduler: elicit pass at tick %d: %w", t, err)
		}
	}

	// Phase 4: snapshot L_cog_ind for every live agent.
	if err := s.snapshotCogInd(t); err != nil {
		return err
	}

	// Phase 5: scenario script ops scheduled to execute *at* tick t.
	s.mu.Lock()
	ops := s.scenarioOps[t]
	delete(s.scenarioOps, t)
	s.mu.Unlock()
	for _, op := range ops {
		if err := op(); err != nil {
			return fmt.Errorf("scheduler: scenario op at tick %d: %w", t, err)
		}
	}

	// Phase 6: advance.
	s.currentTick.Add(1)
	return nil
}

type pendingEvent struct {
	env       *eventsv1.EventEnvelope
	transport AgentTransport
	synthetic bool
}

// commitAgentEnvelopeWithReceipt runs the verify → validate → commit
// pipeline and returns a CommitReceipt suitable for handing back to the
// agent via SubmitEvent. Rejections are returned as receipts with
// committed=false and a rejection_reason; only internal (kernel)
// failures bubble up as Go errors.
func (s *Scheduler) commitAgentEnvelopeWithReceipt(env *eventsv1.EventEnvelope) (*kernelv1.CommitReceipt, error) {
	if env == nil {
		return rejectionReceipt("nil envelope"), nil
	}
	v, err := schema.Validate(env)
	if err != nil {
		return rejectionReceipt(err.Error()), nil
	}
	if err := crypto.VerifyEnvelope(env); err != nil {
		return rejectionReceipt(err.Error()), nil
	}
	committed, err := s.ls.Commit(env, v.Destination)
	if err != nil {
		return rejectionReceipt(err.Error()), nil
	}
	return &kernelv1.CommitReceipt{
		Committed:         true,
		LedgerId:          committed.GetLedgerId(),
		LedgerChannelId:   committed.GetLedgerChannelId(),
		SequencePerLedger: committed.GetEnvelope().GetSequencePerLedger(),
		GlobalCommitSeq:   committed.GetGlobalCommitSeq(),
		ChainHash:         committed.GetChainHash(),
		KernelSignature:   committed.GetKernelSignature(),
	}, nil
}

func rejectionReceipt(reason string) *kernelv1.CommitReceipt {
	return &kernelv1.CommitReceipt{Committed: false, RejectionReason: reason}
}

// synthesizeYield builds a kernel-signed Yield_Kernel_Synthesized
// envelope with the agent's pubkey as emitter (so L_pub readers see
// "agent X yielded this tick"). The signature is by the kernel key,
// not the agent's — consumers distinguish via event_type.
func (s *Scheduler) synthesizeYield(agentPubkey []byte, tick uint64) (*eventsv1.EventEnvelope, error) {
	payload, err := crypto.CanonicalBytes(&eventsv1.Yield{Reason: "tick_timeout"})
	if err != nil {
		return nil, err
	}
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     agentPubkey,
		Tick:              tick,
		SequencePerLedger: s.ls.NextSeqFor(agentPubkey, dest),
		EventType:         "Yield_Kernel_Synthesized",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(s.kernelPriv, env)
	if err != nil {
		return nil, err
	}
	env.Signature = sig
	return env, nil
}

// recordTickTimeout writes a TickTimeoutIncident to L_ctrl. Best-effort:
// commit errors are not propagated (the tick must proceed).
func (s *Scheduler) recordTickTimeout(agentID string, tick uint64) {
	payload, err := crypto.CanonicalBytes(&eventsv1.TickTimeoutIncident{
		AgentId:     agentID,
		Tick:        tick,
		BudgetNanos: uint64(s.tickTimeout.Nanoseconds()),
	})
	if err != nil {
		return
	}
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     s.kernelPub,
		Tick:              tick,
		SequencePerLedger: s.ls.NextSeqFor(s.kernelPub, dest),
		EventType:         "TickTimeoutIncident",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(s.kernelPriv, env)
	if err != nil {
		return
	}
	env.Signature = sig
	_, _ = s.ls.Commit(env, dest)
}

// snapshotCogInd writes one CogIndSnapshot per live agent to L_cog_ind,
// in the canonical agent_id sort order. Returns the first commit error.
func (s *Scheduler) snapshotCogInd(tick uint64) error {
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_COG_IND}
	for _, snap := range s.h.CogSnapshots() {
		payload, err := crypto.CanonicalBytes(&eventsv1.CogIndSnapshot{
			AgentId: snap.AgentID,
			Theta:   snap.Theta,
		})
		if err != nil {
			return err
		}
		env := &eventsv1.EventEnvelope{
			EmitterPubkey:     s.kernelPub,
			Tick:              tick,
			SequencePerLedger: s.ls.NextSeqFor(s.kernelPub, dest),
			EventType:         "CogIndSnapshot",
			Payload:           payload,
		}
		sig, err := crypto.SignEnvelope(s.kernelPriv, env)
		if err != nil {
			return err
		}
		env.Signature = sig
		if _, err := s.ls.Commit(env, dest); err != nil {
			return fmt.Errorf("scheduler: commit L_cog_ind for %s tick %d: %w", snap.AgentID, tick, err)
		}
	}
	return nil
}

// runElicitPass implements scheduler phase 3.5 (Step 4):
//
//   1. For each (agent, action_event) in `pendings` whose action was
//      non-synthetic and committed, in agent_id sorted order, write
//      an ElicitationRequest event to L_ctrl carrying the kernel-
//      rendered action_summary.
//   2. Build an elicit View (copy of the action View shape with
//      `elicit_request` populated and `available_tools` restricted
//      to `[declare_intent]`) and dispatch via the same
//      transport.Tick call as a normal View.
//   3. Receive a DeclareIntent envelope. Commit it through the
//      standard pipeline (schema.Validate routes to L_cog_eli).
//   4. If the transport returns a non-DeclareIntent envelope (Step 2
//      stubs that don't know about elicit), record a
//      MalformedSubmission on L_ctrl with failure="elicit_non_compliance"
//      and skip the L_cog_eli write.
func (s *Scheduler) runElicitPass(ctx context.Context, t uint64, pendings []*pendingEvent) error {
	// Group by agent_id for deterministic order (action commit order
	// is by emitter_pubkey lex; elicit order is by agent_id sorted —
	// the two are usually equivalent under our scenario YAMLs but the
	// sort makes the difference explicit).
	type elicitTarget struct {
		agentID   string
		transport AgentTransport
		envAction *eventsv1.EventEnvelope
	}
	targets := make([]elicitTarget, 0, len(pendings))
	registered := s.h.AllRegisteredAgents()
	for _, p := range pendings {
		if p.synthetic || p.transport == nil {
			// Synthetic-yield agents have no transport (and no
			// reasoning to elicit); kernel-side stubs skip elicit.
			continue
		}
		agentID, ok := agentIDForPubkey(registered, p.env.GetEmitterPubkey())
		if !ok {
			continue
		}
		targets = append(targets, elicitTarget{
			agentID:   agentID,
			transport: p.transport,
			envAction: p.env,
		})
	}
	sort.SliceStable(targets, func(i, j int) bool { return targets[i].agentID < targets[j].agentID })

	for _, tgt := range targets {
		// 1. Write ElicitationRequest to L_ctrl.
		summary, err := elicit.RenderActionSummary(tgt.envAction)
		if err != nil {
			summary = tgt.envAction.GetEventType()
		}
		reqEvent, err := s.commitElicitationRequest(t, tgt.agentID, tgt.envAction, summary)
		if err != nil {
			return err
		}

		// 2. Build elicit View by reusing the action View's shape
		// (channel memberships, scenario, etc.) but with elicit_request
		// populated and tool list restricted to declare_intent.
		v, err := view.BuildView(s.h, s.ls, tgt.agentID, t)
		if err != nil {
			return fmt.Errorf("scheduler: build elicit view for %s tick %d: %w", tgt.agentID, t, err)
		}
		view.ResolveSenderIDs(v, s.h, s.ls)
		v.ElicitRequest = reqEvent
		v.AvailableTools = []string{"declare_intent"}

		tickCtx, cancel := context.WithDeadline(ctx, time.Now().Add(s.tickTimeout))
		respEnv, err := tgt.transport.Tick(tickCtx, v)
		cancel()
		if err != nil || respEnv == nil {
			s.recordMalformedSubmission(tgt.agentID, t, []byte("no_elicit_response"), "elicit_timeout")
			continue
		}

		// 3. Commit only DeclareIntent. Anything else is non-compliance.
		if respEnv.GetEventType() != "DeclareIntent" {
			s.recordMalformedSubmission(
				tgt.agentID, t, []byte(respEnv.GetEventType()), "elicit_non_compliance",
			)
			// Bridge transports need a receipt to unblock the awaiting
			// SubmitEvent goroutine; emit a rejection so the Python
			// side doesn't hang.
			if cn, ok := tgt.transport.(CommitNotifier); ok {
				cn.NotifyCommit(respEnv, rejectionReceipt("elicit_non_compliance"))
			}
			continue
		}
		receipt, err := s.commitAgentEnvelopeWithReceipt(respEnv)
		if err != nil {
			return err
		}
		if cn, ok := tgt.transport.(CommitNotifier); ok {
			cn.NotifyCommit(respEnv, receipt)
		}
	}
	return nil
}

// agentIDForPubkey is a defensive helper for the small registered map.
func agentIDForPubkey(registered map[string]handler.AgentState, pubkey []byte) (string, bool) {
	for id, a := range registered {
		if bytes.Equal(a.Pubkey, pubkey) {
			return id, true
		}
	}
	return "", false
}

// commitElicitationRequest writes an ElicitationRequest event to L_ctrl
// (kernel-signed, kernel as emitter) and returns the payload so the
// View.elicit_request field can carry it to the agent.
func (s *Scheduler) commitElicitationRequest(
	t uint64, agentID string, actionEnv *eventsv1.EventEnvelope, summary string,
) (*eventsv1.ElicitationRequest, error) {
	req := &eventsv1.ElicitationRequest{
		AgentId:                agentID,
		Tick:                   t,
		ActionEventType:        actionEnv.GetEventType(),
		ActionSummary:          summary,
		ActionGlobalCommitSeq:  0, // populated once the action commits — we don't have its global_commit_seq here; left zero for Step 4 (analysis pipeline joins via tick+agent_id).
	}
	payload, err := crypto.CanonicalBytes(req)
	if err != nil {
		return nil, fmt.Errorf("scheduler: marshal ElicitationRequest: %w", err)
	}
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     s.kernelPub,
		Tick:              t,
		SequencePerLedger: s.ls.NextSeqFor(s.kernelPub, dest),
		EventType:         "ElicitationRequest",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(s.kernelPriv, env)
	if err != nil {
		return nil, err
	}
	env.Signature = sig
	if _, err := s.ls.Commit(env, dest); err != nil {
		return nil, fmt.Errorf("scheduler: commit ElicitationRequest: %w", err)
	}
	return req, nil
}

// recordMalformedSubmission writes a MalformedSubmission event to
// L_ctrl. Best-effort — commit errors don't propagate (the tick must
// continue).
func (s *Scheduler) recordMalformedSubmission(
	agentID string, t uint64, rawOutput []byte, failure string,
) {
	payload, err := crypto.CanonicalBytes(&eventsv1.MalformedSubmission{
		AgentId:   agentID,
		Tick:      t,
		RawOutput: rawOutput,
		Failure:   failure,
	})
	if err != nil {
		return
	}
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     s.kernelPub,
		Tick:              t,
		SequencePerLedger: s.ls.NextSeqFor(s.kernelPub, dest),
		EventType:         "MalformedSubmission",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(s.kernelPriv, env)
	if err != nil {
		return
	}
	env.Signature = sig
	_, _ = s.ls.Commit(env, dest)
}
