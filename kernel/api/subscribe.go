package api

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	kernelv1 "github.com/byzminds/byzminds/proto/kernelv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

// AttachStep2 wires the Step 2 components (scheduler, handler) onto the
// gRPC server, enabling Subscribe and routing SubmitEvent for registered
// agents through the scheduler's per-agent inbox. Step 1's
// SubmitEvent path (direct commit) remains the fallback for non-agent
// callers (researcher control envelopes, etc.).
func (s *Server) AttachStep2(h *handler.Handler, sch *scheduler.Scheduler) {
	s.handler = h
	s.scheduler = sch
	if s.bridges == nil {
		s.bridges = make(map[string]*subscriberBridge)
	}
}

// subscriberBridge is the per-agent in-flight gRPC stream state.
//
// SubmitEvent pushes a submission onto `inbox`; the scheduler's
// Tick() reads it, stores it on `pendingSub`, and returns the env to
// the scheduler. After the scheduler commits in (emitter_pubkey lex)
// order, it calls NotifyCommit, which sends the receipt back through
// the stored submission's receipt channel — completing the awaiting
// SubmitEvent goroutine.
type subscriberBridge struct {
	agentID    string
	views      chan *viewv1.View
	inbox      chan submission
	done       chan struct{}
	pendingMu  sync.Mutex
	pendingSub *submission
}

type submission struct {
	env     *eventsv1.EventEnvelope
	receipt chan *kernelv1.CommitReceipt
}

// Subscribe implements kernelv1.KernelServer. Verifies the agent's
// signature, registers a bridge under the agent's id, and pumps views
// from the bridge channel until the client closes the stream.
//
// Step 2 attachment: AttachStep2 must have been called for Subscribe
// to function. Without the scheduler attached we return Unavailable.
func (s *Server) Subscribe(req *kernelv1.SubscribeRequest, stream grpc.ServerStreamingServer[viewv1.View]) error {
	if s.handler == nil || s.scheduler == nil {
		return errors.New("api: Subscribe unavailable (kernel not in Step 2 mode)")
	}
	if len(req.GetAgentPubkey()) != crypto.PublicKeySize {
		return fmt.Errorf("api: agent_pubkey must be %d bytes", crypto.PublicKeySize)
	}
	if len(req.GetSignature()) != crypto.SignatureSize {
		return fmt.Errorf("api: signature must be %d bytes", crypto.SignatureSize)
	}
	if err := crypto.VerifyBytes(req.GetAgentPubkey(), subscribeSigningBytes(req), req.GetSignature()); err != nil {
		return fmt.Errorf("api: subscribe signature invalid: %w", err)
	}
	agentID, ok := s.handler.LookupAgentIDByPubkey(req.GetAgentPubkey())
	if !ok {
		return fmt.Errorf("api: pubkey is not bound to any spawned agent")
	}

	br := &subscriberBridge{
		agentID: agentID,
		views:   make(chan *viewv1.View, 4),
		inbox:   make(chan submission, 4),
		done:    make(chan struct{}),
	}
	s.bridgesMu.Lock()
	if _, dup := s.bridges[agentID]; dup {
		s.bridgesMu.Unlock()
		return fmt.Errorf("api: agent %q already has an active subscription", agentID)
	}
	s.bridges[agentID] = br
	s.bridgesMu.Unlock()

	if err := s.scheduler.AttachAgent(agentID, br); err != nil {
		s.bridgesMu.Lock()
		delete(s.bridges, agentID)
		s.bridgesMu.Unlock()
		return fmt.Errorf("api: attach to scheduler: %w", err)
	}

	defer func() {
		s.bridgesMu.Lock()
		delete(s.bridges, agentID)
		s.bridgesMu.Unlock()
		close(br.done)
	}()

	for {
		select {
		case <-stream.Context().Done():
			return stream.Context().Err()
		case v := <-br.views:
			if err := stream.Send(v); err != nil {
				return err
			}
		}
	}
}

// Tick implements scheduler.AgentTransport via the bridge: pushes the
// view onto the stream, waits for an envelope on the inbox (or
// deadline), stores the submission so NotifyCommit can later resolve
// the SubmitEvent receipt.
func (br *subscriberBridge) Tick(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	select {
	case br.views <- v:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case sub := <-br.inbox:
		br.pendingMu.Lock()
		br.pendingSub = &sub
		br.pendingMu.Unlock()
		return sub.env, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// NotifyCommit implements scheduler.CommitNotifier. Called by the
// scheduler after it commits (or rejects) the envelope handed back by
// Tick. The receipt is sent on the awaiting submission's receipt
// channel; if no submission is pending (e.g., the scheduler synthesized
// the envelope on timeout) the call is a no-op.
func (br *subscriberBridge) NotifyCommit(_ *eventsv1.EventEnvelope, rec *kernelv1.CommitReceipt) {
	br.pendingMu.Lock()
	sub := br.pendingSub
	br.pendingSub = nil
	br.pendingMu.Unlock()
	if sub == nil {
		return
	}
	select {
	case sub.receipt <- rec:
	default:
	}
}

// subscribeSigningBytes is the canonical preimage the agent signs.
func subscribeSigningBytes(req *kernelv1.SubscribeRequest) []byte {
	// reader_pubkey || from_tick big-endian, mirroring ViewRequest.
	return ViewRequestSigningBytes(req.GetAgentPubkey(), req.GetFromTick())
}
