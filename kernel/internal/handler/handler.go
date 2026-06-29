package handler

import (
	"crypto/ed25519"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

// ThetaDials is the fixed length of the bias vector (six dials per the
// state summary: authority, bandwagon, sycophancy, free_ride, collude,
// deceive). Stage A baselines write zeros.
const ThetaDials = 6

// DefaultPublicHistoryWindow / DefaultPrivateHistoryWindow are the K_c
// fallbacks per the template spec §8.
const (
	DefaultPublicHistoryWindow  uint32 = 20
	DefaultPrivateHistoryWindow uint32 = 10
)

// Handler is the world-state owner.
type Handler struct {
	ls    *ledger.LedgerSet
	sched SchedulerControl // set via SetScheduler after scheduler is built
	st    *state

	kernelPub  ed25519.PublicKey
	kernelPriv ed25519.PrivateKey
	researcher ed25519.PublicKey
}

// New constructs a Handler bound to ls. Scheduler must be attached via
// SetScheduler before any tick-aware method is called.
func New(ls *ledger.LedgerSet) *Handler {
	priv := ls.KernelPriv()
	// Derive the kernel public key from the private key.
	pub := priv.Public().(ed25519.PublicKey)
	return &Handler{
		ls:         ls,
		st:         newState(),
		kernelPub:  pub,
		kernelPriv: priv,
		researcher: ls.Researcher(),
	}
}

// SetScheduler attaches the scheduler controller. Must be called once
// before any tick-aware method runs.
func (h *Handler) SetScheduler(s SchedulerControl) { h.sched = s }

// ResearcherPubkey returns the researcher pubkey used to authenticate
// inbound HandlerRPC calls.
func (h *Handler) ResearcherPubkey() ed25519.PublicKey { return h.researcher }

// LoadScenario installs the scenario state. Called once by the scenario
// loader before the scheduler starts ticking. It writes a "LoadScenario"
// control event to L_ctrl recording the scenario name + YAML hash.
func (h *Handler) LoadScenario(s *ScenarioState) error {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if h.st.scenario != nil {
		return errors.New("handler: scenario already loaded")
	}
	if s.HistoryWindow == nil {
		s.HistoryWindow = map[string]uint32{}
	}
	if _, ok := s.HistoryWindow["public"]; !ok {
		s.HistoryWindow["public"] = DefaultPublicHistoryWindow
	}
	if _, ok := s.HistoryWindow["private"]; !ok {
		s.HistoryWindow["private"] = DefaultPrivateHistoryWindow
	}
	// Canonicalize phase tool lists.
	for i := range s.Phases {
		sort.Strings(s.Phases[i].AvailableTools)
	}
	h.st.scenario = s
	_, err := h.recordControlEventLocked("LoadScenario", &handlerv1.SpawnAgentRequest{}, 0, []byte(s.YAMLHash))
	return err
}

// -----------------------------------------------------------------------
// Mutation methods. Each:
//   1. Authenticates the request via verifyAuth (no-op for in-process
//      callers that supply a nil auth; scenario loader uses this path).
//   2. Computes effective_tick.
//   3. Mutates or queues state.
//   4. Writes the HandlerControlEvent to L_ctrl.
//   5. Returns the ack.
// -----------------------------------------------------------------------

// SpawnAgent registers a new agent. Effective immediately at currentTick.
func (h *Handler) SpawnAgent(req *handlerv1.SpawnAgentRequest) (*handlerv1.SpawnAgentResponse, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	if req.GetAgentId() == "" {
		return nil, errors.New("handler: SpawnAgent.agent_id required")
	}
	if len(req.GetAgentPubkey()) != crypto.PublicKeySize {
		return nil, fmt.Errorf("handler: SpawnAgent.agent_pubkey must be %d bytes", crypto.PublicKeySize)
	}
	theta := req.GetTheta()
	if len(theta) == 0 {
		theta = make([]float64, ThetaDials)
	}
	if len(theta) != ThetaDials {
		return nil, fmt.Errorf("handler: SpawnAgent.theta must have length %d", ThetaDials)
	}

	tick := h.currentTickLocked()

	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if _, dup := h.st.agents[req.GetAgentId()]; dup {
		return nil, fmt.Errorf("handler: agent_id %q already spawned", req.GetAgentId())
	}
	pkHex := hex.EncodeToString(req.GetAgentPubkey())
	if existing, dup := h.st.pubkeyToID[pkHex]; dup {
		return nil, fmt.Errorf("handler: pubkey already bound to agent %q", existing)
	}
	thetaCopy := append([]float64(nil), theta...)
	h.st.agents[req.GetAgentId()] = &AgentState{
		ID:         req.GetAgentId(),
		Pubkey:     append([]byte(nil), req.GetAgentPubkey()...),
		Role:       req.GetRole(),
		StubPolicy: req.GetStubPolicy(),
		Theta:      thetaCopy,
		SpawnTick:  tick,
		alive:      true,
	}
	h.st.pubkeyToID[pkHex] = req.GetAgentId()
	h.st.caps[req.GetAgentId()] = &CapabilityState{Loaded: map[string]bool{}}
	h.st.rebuildLiveOrderingLocked()

	if _, err := h.recordControlEventLocked("SpawnAgent", req, tick, nil); err != nil {
		return nil, err
	}
	return &handlerv1.SpawnAgentResponse{AgentId: req.GetAgentId(), SpawnTick: tick}, nil
}

// KillAgent removes an agent from the live roster at tick+1.
func (h *Handler) KillAgent(req *handlerv1.KillAgentRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	if req.GetAgentId() == "" {
		return nil, errors.New("handler: KillAgent.agent_id required")
	}
	tick := h.currentTickLocked()
	effective := tick + 1

	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	ag, ok := h.st.agents[req.GetAgentId()]
	if !ok || !ag.alive {
		return nil, fmt.Errorf("handler: agent %q is not alive", req.GetAgentId())
	}
	h.st.queuedAt[effective] = append(h.st.queuedAt[effective], queuedOp{
		label: "KillAgent:" + req.GetAgentId(),
		apply: func() error {
			ag.alive = false
			ag.KilledAt = effective
			h.st.rebuildLiveOrderingLocked()
			return nil
		},
	})
	receipt, err := h.recordControlEventLocked("KillAgent", req, effective, nil)
	if err != nil {
		return nil, err
	}
	return receipt, nil
}

// Retune queues a theta replacement for tick+1. The new value is the
// first one snapshotted to L_cog_ind at tick+1.
func (h *Handler) Retune(req *handlerv1.RetuneRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	if len(req.GetTheta()) != ThetaDials {
		return nil, fmt.Errorf("handler: Retune.theta must have length %d", ThetaDials)
	}
	tick := h.currentTickLocked()
	effective := tick + 1

	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	ag, ok := h.st.agents[req.GetAgentId()]
	if !ok || !ag.alive {
		return nil, fmt.Errorf("handler: agent %q is not alive", req.GetAgentId())
	}
	newTheta := append([]float64(nil), req.GetTheta()...)
	h.st.queuedAt[effective] = append(h.st.queuedAt[effective], queuedOp{
		label: "Retune:" + req.GetAgentId(),
		apply: func() error {
			ag.Theta = newTheta
			return nil
		},
	})
	return h.recordControlEventLocked("Retune", req, effective, nil)
}

// OpenChannel (handler-initiated) mints L_prv[channel_id] immediately
// and marks the listed agent_ids as members. Effective at currentTick.
func (h *Handler) OpenChannel(req *handlerv1.OpenChannelRequest) (*handlerv1.OpenChannelResponse, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	if len(req.GetMemberAgentIds()) < 2 {
		return nil, errors.New("handler: OpenChannel requires >= 2 members")
	}
	tick := h.currentTickLocked()
	return h.openChannelAt(req, tick)
}

// openChannelAt is the inner mint routine, shared by the explicit
// OpenChannel RPC (effective immediately) and the agent-initiated
// auto-approval path (effective at tick+1).
func (h *Handler) openChannelAt(req *handlerv1.OpenChannelRequest, effective uint64) (*handlerv1.OpenChannelResponse, error) {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()

	channelID := req.GetChannelId()
	if channelID == "" {
		channelID = h.allocateChannelIDLocked()
	}
	if _, dup := h.st.channels[channelID]; dup {
		return nil, fmt.Errorf("handler: channel %q already open", channelID)
	}

	members := append([]string(nil), req.GetMemberAgentIds()...)
	sort.Strings(members)
	memberPubkeys, err := h.memberPubkeysLocked(members)
	if err != nil {
		return nil, err
	}

	if err := h.ls.OpenPrivateChannel(channelID, memberPubkeys); err != nil {
		return nil, fmt.Errorf("handler: OpenPrivateChannel: %w", err)
	}
	h.st.channels[channelID] = &ChannelState{
		ChannelID: channelID,
		Members:   members,
		OpenedAt:  effective,
	}

	if _, err := h.recordControlEventLocked("OpenChannel", req, effective, []byte(channelID)); err != nil {
		return nil, err
	}
	return &handlerv1.OpenChannelResponse{ChannelId: channelID, OpenTick: effective}, nil
}

// CloseChannel freezes a private channel at tick+1.
func (h *Handler) CloseChannel(req *handlerv1.CloseChannelRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	tick := h.currentTickLocked()
	effective := tick + 1

	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	ch, ok := h.st.channels[req.GetChannelId()]
	if !ok {
		return nil, fmt.Errorf("handler: unknown channel %q", req.GetChannelId())
	}
	if ch.ClosedAt != 0 {
		return nil, fmt.Errorf("handler: channel %q already closed", req.GetChannelId())
	}
	h.st.queuedAt[effective] = append(h.st.queuedAt[effective], queuedOp{
		label: "CloseChannel:" + req.GetChannelId(),
		apply: func() error {
			ch.ClosedAt = effective
			h.ls.FreezePrivateChannel(ch.ChannelID)
			return nil
		},
	})
	return h.recordControlEventLocked("CloseChannel", req, effective, nil)
}

// AssignTask records the task payload + kind for each named agent.
// Effective immediately at currentTick.
func (h *Handler) AssignTask(req *handlerv1.AssignTaskRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	tick := h.currentTickLocked()
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	for _, id := range req.GetAgentIds() {
		ag, ok := h.st.agents[id]
		if !ok || !ag.alive {
			return nil, fmt.Errorf("handler: agent %q not alive", id)
		}
		h.st.taskAssignments[id] = append([]byte(nil), req.GetTaskBlob()...)
		h.st.taskKinds[id] = req.GetTaskKind()
	}
	return h.recordControlEventLocked("AssignTask", req, tick, nil)
}

// InjectExternalMessage queues an external message for the agent's
// next BuildView call. Effective immediately at currentTick.
func (h *Handler) InjectExternalMessage(req *handlerv1.InjectExternalMessageRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	if req.GetAgentId() == "" {
		return nil, errors.New("handler: Inject.agent_id required")
	}
	tick := h.currentTickLocked()
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if _, ok := h.st.agents[req.GetAgentId()]; !ok {
		return nil, fmt.Errorf("handler: agent %q not registered", req.GetAgentId())
	}
	h.st.externals[req.GetAgentId()] = append(h.st.externals[req.GetAgentId()], PendingExternal{
		ClaimedSource: req.GetClaimedSource(),
		Content:       req.GetContent(),
		InjectTick:    tick,
	})
	return h.recordControlEventLocked("InjectExternalMessage", req, tick, nil)
}

// Pause asks the scheduler to halt at the next tick boundary.
func (h *Handler) Pause(req *handlerv1.PauseRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	tick := h.currentTickLocked()
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if h.sched != nil {
		h.sched.Pause()
	}
	return h.recordControlEventLocked("Pause", req, tick, nil)
}

// Resume releases the scheduler to advance freely.
func (h *Handler) Resume(req *handlerv1.ResumeRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	tick := h.currentTickLocked()
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if h.sched != nil {
		h.sched.Resume()
	}
	return h.recordControlEventLocked("Resume", req, tick, nil)
}

// Step advances the scheduler by n tick boundaries (n=0 ⇒ 1).
func (h *Handler) Step(req *handlerv1.StepRequest) (*handlerv1.HandlerAck, error) {
	if err := h.verifyAuth(req.GetAuth(), req); err != nil {
		return nil, err
	}
	n := req.GetTicks()
	if n == 0 {
		n = 1
	}
	tick := h.currentTickLocked()
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if h.sched != nil {
		h.sched.Step(n)
	}
	return h.recordControlEventLocked("Step", req, tick, nil)
}

// -----------------------------------------------------------------------
// Tick-loop hooks (called by the scheduler)
// -----------------------------------------------------------------------

// DrainQueueAt applies all queued mutations scheduled for `tick`. The
// scheduler invokes this at the start of each tick, before any view
// dispatch or L_cog_ind snapshot, so deferred Retune values are in place
// before the snapshot reads them.
func (h *Handler) DrainQueueAt(tick uint64) error {
	h.st.mu.Lock()
	ops := h.st.queuedAt[tick]
	delete(h.st.queuedAt, tick)
	h.st.mu.Unlock()
	for _, op := range ops {
		if err := op.apply(); err != nil {
			return fmt.Errorf("handler: queued op %q: %w", op.label, err)
		}
	}
	return nil
}

// HandleAgentControlEvent reacts to agent-emitted OpenChannelReq /
// CloseChannelReq events that the scheduler committed to L_ctrl. The
// kernel's Stage A policy: auto-approve OpenChannel iff the emitter is
// in proposed_members, schedule mint for currentTick + 1. CloseChannel
// is auto-approved if the emitter is a current member.
func (h *Handler) HandleAgentControlEvent(env *eventsv1.EventEnvelope, currentTick uint64) error {
	switch env.GetEventType() {
	case "OpenChannelReq":
		msg := &eventsv1.OpenChannelReq{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return fmt.Errorf("handler: parse OpenChannelReq: %w", err)
		}
		emitterID, ok := h.LookupAgentIDByPubkey(env.GetEmitterPubkey())
		if !ok {
			return nil // unregistered emitter; ignore (kernel committed nothing actionable)
		}
		members := msg.GetProposedMembers()
		if !containsString(members, emitterID) {
			return nil // policy: emitter must be in proposed_members
		}
		effective := currentTick + 1
		auth := &handlerv1.HandlerAuth{CallerPubkey: h.researcher}
		openReq := &handlerv1.OpenChannelRequest{
			Auth:             auth,
			MemberAgentIds:   members,
		}
		h.st.mu.Lock()
		h.st.queuedAt[effective] = append(h.st.queuedAt[effective], queuedOp{
			label: "AutoOpenChannel",
			apply: func() error {
				// Best-effort: an agent (especially an LLM) may propose a
				// non-existent member or an otherwise invalid open. Such a
				// request must not crash the panel -- the OpenChannelReq is
				// already recorded on L_ctrl, so the attempt is preserved; we
				// simply decline to mint the channel.
				_, _ = h.openChannelAt(openReq, effective)
				return nil
			},
		})
		h.st.mu.Unlock()
		return nil
	case "CloseChannelReq":
		// Step 2 policy: auto-approve if the emitter is in the channel.
		msg := &eventsv1.CloseChannelReq{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return fmt.Errorf("handler: parse CloseChannelReq: %w", err)
		}
		emitterID, ok := h.LookupAgentIDByPubkey(env.GetEmitterPubkey())
		if !ok {
			return nil
		}
		ch, ok := h.LookupChannel(msg.GetChannelId())
		if !ok || !containsString(ch.Members, emitterID) {
			return nil
		}
		closeReq := &handlerv1.CloseChannelRequest{
			Auth:      &handlerv1.HandlerAuth{CallerPubkey: h.researcher},
			ChannelId: msg.GetChannelId(),
		}
		if _, err := h.CloseChannel(closeReq); err != nil {
			return err
		}
		return nil
	default:
		return nil
	}
}

// -----------------------------------------------------------------------
// Read accessors used by the view builder + scheduler.
// -----------------------------------------------------------------------

// LookupAgentIDByPubkey resolves a SubmitEvent envelope back to its
// registered agent_id. Returns false if no registered agent owns this
// pubkey (e.g., the researcher submitting a control envelope).
func (h *Handler) LookupAgentIDByPubkey(pubkey []byte) (string, bool) {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	id, ok := h.st.pubkeyToID[hex.EncodeToString(pubkey)]
	return id, ok
}

// LookupAgent returns a defensive copy of one AgentState (or false).
func (h *Handler) LookupAgent(agentID string) (AgentState, bool) {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	a, ok := h.st.agents[agentID]
	if !ok {
		return AgentState{}, false
	}
	return *a, true
}

// LookupChannel returns a defensive copy of one ChannelState.
func (h *Handler) LookupChannel(channelID string) (ChannelState, bool) {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	c, ok := h.st.channels[channelID]
	if !ok {
		return ChannelState{}, false
	}
	out := *c
	out.Members = append([]string(nil), c.Members...)
	return out, true
}

// LiveAgents returns the sorted list of currently-alive agent_ids.
func (h *Handler) LiveAgents() []string {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	out := append([]string(nil), h.st.liveOrdering...)
	return out
}

// AgentChannelMemberships returns the sorted list of channel_ids that
// `agentID` belongs to (private only — "public" is implicit and added
// by the view builder).
func (h *Handler) AgentChannelMemberships(agentID string) []string {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	var out []string
	for cid, ch := range h.st.channels {
		if ch.ClosedAt != 0 {
			continue
		}
		if containsString(ch.Members, agentID) {
			out = append(out, cid)
		}
	}
	sort.Strings(out)
	return out
}

// AgentLoadedCapabilities returns sorted cap_ids loaded on agentID.
func (h *Handler) AgentLoadedCapabilities(agentID string) []string {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	c, ok := h.st.caps[agentID]
	if !ok {
		return nil
	}
	out := make([]string, 0, len(c.Loaded))
	for k, v := range c.Loaded {
		if v {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// Scenario returns the loaded scenario (or nil).
func (h *Handler) Scenario() *ScenarioState {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	return h.st.scenario
}

// PhaseAt returns the phase name, 1-indexed round, and total rounds for
// the given tick. Tick 0 is the first round of the first phase.
func (h *Handler) PhaseAt(tick uint64) (string, uint32, uint32, []string) {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	if h.st.scenario == nil {
		return "", 0, 0, nil
	}
	cum := uint64(0)
	for _, p := range h.st.scenario.Phases {
		if tick < cum+uint64(p.Rounds) {
			round := uint32(tick - cum + 1)
			tools := append([]string(nil), p.AvailableTools...)
			return p.Name, round, p.Rounds, tools
		}
		cum += uint64(p.Rounds)
	}
	// After all scheduled phases: stay in the final phase, round 0 sentinel.
	last := h.st.scenario.Phases[len(h.st.scenario.Phases)-1]
	tools := append([]string(nil), last.AvailableTools...)
	return last.Name, 0, last.Rounds, tools
}

// DrainExternalInjects returns + clears the pending externals for agentID.
func (h *Handler) DrainExternalInjects(agentID string) []PendingExternal {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	out := h.st.externals[agentID]
	h.st.externals[agentID] = nil
	return out
}

// AllAgentsForCogSnapshot returns the live agent_ids and their current
// theta. The scheduler uses this to write L_cog_ind at each tick.
func (h *Handler) AllAgentsForCogSnapshot() []cogSnapshotEntry {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	out := make([]cogSnapshotEntry, 0, len(h.st.liveOrdering))
	for _, id := range h.st.liveOrdering {
		ag := h.st.agents[id]
		out = append(out, cogSnapshotEntry{
			AgentID: id,
			Theta:   append([]float64(nil), ag.Theta...),
		})
	}
	return out
}

type cogSnapshotEntry struct {
	AgentID string
	Theta   []float64
}

// CogSnapshot is the exported type the scheduler iterates over.
type CogSnapshot struct {
	AgentID string
	Theta   []float64
}

// CogSnapshots converts the internal type to its exported form. Cheap copy.
func (h *Handler) CogSnapshots() []CogSnapshot {
	in := h.AllAgentsForCogSnapshot()
	out := make([]CogSnapshot, len(in))
	for i, e := range in {
		out[i] = CogSnapshot{AgentID: e.AgentID, Theta: e.Theta}
	}
	return out
}

// AllRegisteredAgents returns every agent (alive + killed), keyed by
// agent_id, with pubkey + role. Used by the scheduler when routing
// SubmitEvent calls back to the right inbox.
func (h *Handler) AllRegisteredAgents() map[string]AgentState {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	out := make(map[string]AgentState, len(h.st.agents))
	for id, a := range h.st.agents {
		out[id] = *a
	}
	return out
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func (h *Handler) currentTickLocked() uint64 {
	if h.sched == nil {
		return 0
	}
	return h.sched.CurrentTick()
}

func (h *Handler) memberPubkeysLocked(memberIDs []string) ([]ledger.Identity, error) {
	out := make([]ledger.Identity, 0, len(memberIDs))
	for _, id := range memberIDs {
		ag, ok := h.st.agents[id]
		if !ok || !ag.alive {
			return nil, fmt.Errorf("handler: cannot open channel — agent %q not alive", id)
		}
		out = append(out, append(ledger.Identity{}, ag.Pubkey...))
	}
	return out, nil
}

func (h *Handler) allocateChannelIDLocked() string {
	for i := 1; ; i++ {
		id := fmt.Sprintf("ch_auto_%d", i)
		if _, dup := h.st.channels[id]; !dup {
			return id
		}
	}
}

// recordControlEventLocked builds a HandlerControlEvent envelope, signs
// it with the kernel key, and commits it to L_ctrl. Returns the ack with
// L_ctrl commit metadata. Held: state.mu (so the surrounding mutation
// and the L_ctrl write are atomic to scheduler observers).
//
// extraTag is appended to the recorded handler_request_bytes preimage to
// distinguish auto-allocations (e.g., kernel-assigned channel_id) from
// the caller's bytes — empty for the normal path.
func (h *Handler) recordControlEventLocked(rpcName string, req proto.Message, effectiveTick uint64, extraTag []byte) (*handlerv1.HandlerAck, error) {
	preimage, err := canonicalRequestBytes(req)
	if err != nil {
		return nil, fmt.Errorf("handler: canonicalize %s request: %w", rpcName, err)
	}
	if len(extraTag) > 0 {
		preimage = append(preimage, '|')
		preimage = append(preimage, extraTag...)
	}
	ctrlPayload, err := crypto.CanonicalBytes(&eventsv1.HandlerControlEvent{
		HandlerRpcName:      rpcName,
		HandlerRequestBytes: preimage,
		EffectiveTick:       effectiveTick,
	})
	if err != nil {
		return nil, err
	}
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}
	seq := h.ls.NextSeqFor(h.kernelPub, dest)
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     h.kernelPub,
		Tick:              h.currentTickLocked(),
		SequencePerLedger: seq,
		EventType:         "Handler_" + rpcName,
		Payload:           ctrlPayload,
	}
	sig, err := crypto.SignEnvelope(h.kernelPriv, env)
	if err != nil {
		return nil, err
	}
	env.Signature = sig
	committed, err := h.ls.Commit(env, dest)
	if err != nil {
		return nil, fmt.Errorf("handler: commit L_ctrl record: %w", err)
	}
	return &handlerv1.HandlerAck{
		ControlGlobalCommitSeq: committed.GetGlobalCommitSeq(),
		LCtrlChainHash:         committed.GetChainHash(),
		EffectiveTick:          effectiveTick,
	}, nil
}

// verifyAuth validates the HandlerAuth on a request. For in-process
// callers (the scenario loader), req may carry a nil/zero auth — we
// accept it as trusted, matching the brief's "Hard-coded in the API
// layer" wording (the gRPC wrapper layer is the only one that enforces
// signature checks; the in-process API is trusted).
func (h *Handler) verifyAuth(auth *handlerv1.HandlerAuth, req proto.Message) error {
	if auth == nil {
		return nil
	}
	if len(auth.GetCallerPubkey()) == 0 && len(auth.GetSignature()) == 0 {
		return nil
	}
	if !equalBytes(auth.GetCallerPubkey(), h.researcher) {
		return errors.New("handler: caller_pubkey is not the researcher")
	}
	preimage, err := canonicalRequestBytes(req)
	if err != nil {
		return err
	}
	return crypto.VerifyBytes(h.researcher, preimage, auth.GetSignature())
}

func containsString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- Tests use these to inspect state without a tick advance ---

// MustViewMessageFromCommitted projects a committed Speak envelope into
// the view's Message type. Exported because the view builder uses it.
func MustViewMessageFromCommitted(c *ledgerv1.CommittedEvent, senderID string) *viewv1.Message {
	msg := &eventsv1.Speak{}
	_ = proto.Unmarshal(c.GetEnvelope().GetPayload(), msg)
	return &viewv1.Message{
		SenderId:        senderID,
		Content:         msg.GetContent(),
		Tick:            c.GetEnvelope().GetTick(),
		GlobalCommitSeq: c.GetGlobalCommitSeq(),
	}
}
