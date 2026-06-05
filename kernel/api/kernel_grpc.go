// Package api wires the kernel's internal packages to the gRPC surface
// defined in proto/kernel.proto.
//
// Submit pipeline (synchronous, behind the kernel's commit lock via
// LedgerSet.Commit):
//
//   client → SubmitEvent(envelope)
//      ↓ envelope shape + payload validation (schema.Validate)
//      ↓ Ed25519 verify (crypto.VerifyEnvelope)
//      ↓ destination resolution (schema → ledger.Destination)
//      ↓ commit + chain extension (LedgerSet.Commit)
//      ↓ build receipt with chain head + kernel signature
//   ← CommitReceipt
//
// Rejections at any step are non-recoverable and surfaced as receipts
// with rejection_reason set; per the design, the malformed-submission
// itself is logged to L_ctrl by the dispatcher in Step 2. Step 1 keeps
// it simple: rejection ⇒ receipt with reason, no L_ctrl write.
//
// GetView is a server-streaming RPC. The reader's signature over
// (reader_pubkey || from_tick) is verified, then the merged commit log
// is streamed in (tick, ledger_id, sequence_per_ledger) lex order
// filtered by per-ledger access policy. Step 1 yields the snapshot at
// call time and closes the stream; live tailing arrives in Step 2.
package api

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"sync"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/schema"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	kernelv1 "github.com/byzminds/byzminds/proto/kernelv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// Server implements kernelv1.KernelServer.
type Server struct {
	kernelv1.UnimplementedKernelServer
	ls *ledger.LedgerSet

	// Step 2 attachment. Nil until AttachStep2 is called.
	handler   *handler.Handler
	scheduler *scheduler.Scheduler
	bridges   map[string]*subscriberBridge
	bridgesMu sync.Mutex
}

// New returns a Server bound to ls.
func New(ls *ledger.LedgerSet) *Server { return &Server{ls: ls} }

// Register installs s on grpcServer.
func (s *Server) Register(grpcServer *grpc.Server) {
	kernelv1.RegisterKernelServer(grpcServer, s)
}

// SubmitEvent has two paths:
//
//   1. Subscribed-agent path (Step 2+): if the emitter pubkey matches a
//      registered agent with an active Subscribe bridge, we push the
//      envelope onto the bridge's inbox so the scheduler picks it up
//      during the current tick's collection phase. The scheduler then
//      commits in deterministic (emitter_pubkey lex) order and
//      notifies the bridge with a CommitReceipt, which we relay back.
//
//   2. Direct path (Step 1 fallback): for any other caller (researcher
//      control envelopes, smoke tests submitting from outside the
//      scheduler-bound flow), we run the verify → validate → commit
//      pipeline inline and return a receipt.
func (s *Server) SubmitEvent(ctx context.Context, env *eventsv1.EventEnvelope) (*kernelv1.CommitReceipt, error) {
	if env == nil {
		return reject("nil envelope"), nil
	}
	// Path 1 — route through subscriber bridge if applicable.
	if s.handler != nil {
		if agentID, ok := s.handler.LookupAgentIDByPubkey(env.GetEmitterPubkey()); ok {
			s.bridgesMu.Lock()
			br, hasBridge := s.bridges[agentID]
			s.bridgesMu.Unlock()
			if hasBridge {
				sub := submission{
					env:     env,
					receipt: make(chan *kernelv1.CommitReceipt, 1),
				}
				select {
				case br.inbox <- sub:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				select {
				case rec := <-sub.receipt:
					return rec, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
		}
	}
	// Path 2 — direct commit (Step 1 fallback).
	return s.directSubmit(env), nil
}

// directSubmit runs the verify → validate → commit pipeline inline.
// Used by SubmitEvent for non-subscribed callers; the Step 1
// regression tests exercise this exclusively.
func (s *Server) directSubmit(env *eventsv1.EventEnvelope) *kernelv1.CommitReceipt {
	v, err := schema.Validate(env)
	if err != nil {
		return reject(err.Error())
	}
	if err := crypto.VerifyEnvelope(env); err != nil {
		return reject(err.Error())
	}
	committed, err := s.ls.Commit(env, v.Destination)
	if err != nil {
		return reject(err.Error())
	}
	return &kernelv1.CommitReceipt{
		Committed:         true,
		LedgerId:          committed.GetLedgerId(),
		LedgerChannelId:   committed.GetLedgerChannelId(),
		SequencePerLedger: committed.GetEnvelope().GetSequencePerLedger(),
		GlobalCommitSeq:   committed.GetGlobalCommitSeq(),
		ChainHash:         committed.GetChainHash(),
		KernelSignature:   committed.GetKernelSignature(),
	}
}

func reject(reason string) *kernelv1.CommitReceipt {
	return &kernelv1.CommitReceipt{Committed: false, RejectionReason: reason}
}

// GetView authenticates the reader, snapshots the merged commit log,
// filters it against per-ledger access policy, and streams matches in
// (tick, ledger_id, sequence) lex order from from_tick onward.
func (s *Server) GetView(req *kernelv1.ViewRequest, stream grpc.ServerStreamingServer[kernelv1.EventView]) error {
	if req == nil {
		return errors.New("api: nil ViewRequest")
	}
	if len(req.GetReaderPubkey()) != crypto.PublicKeySize {
		return fmt.Errorf("api: reader_pubkey must be %d bytes", crypto.PublicKeySize)
	}
	if len(req.GetSignature()) != crypto.SignatureSize {
		return fmt.Errorf("api: signature must be %d bytes", crypto.SignatureSize)
	}
	if err := crypto.VerifyBytes(req.GetReaderPubkey(), viewRequestSigningBytes(req), req.GetSignature()); err != nil {
		return fmt.Errorf("api: view request signature invalid: %w", err)
	}

	log := s.ls.CommittedLog()
	// Total order: (tick, ledger_id, global_commit_seq). global_commit_seq
	// is unique across the whole run, so it disambiguates within a
	// (tick, ledger) bucket where per-emitter sequence_per_ledger values
	// can collide between emitters.
	sort.SliceStable(log, func(i, j int) bool {
		ai, aj := log[i], log[j]
		ti, tj := ai.GetEnvelope().GetTick(), aj.GetEnvelope().GetTick()
		if ti != tj {
			return ti < tj
		}
		li, lj := int32(ai.GetLedgerId()), int32(aj.GetLedgerId())
		if li != lj {
			return li < lj
		}
		return ai.GetGlobalCommitSeq() < aj.GetGlobalCommitSeq()
	})

	for _, c := range log {
		if c.GetEnvelope().GetTick() < req.GetFromTick() {
			continue
		}
		if !s.canRead(req.GetReaderPubkey(), c) {
			continue
		}
		if err := stream.Send(&kernelv1.EventView{Event: c}); err != nil {
			return err
		}
	}
	return nil
}

// canRead consults the destination ledger's access policy.
func (s *Server) canRead(reader []byte, c *ledgerv1.CommittedEvent) bool {
	switch c.GetLedgerId() {
	case ledgerv1.LedgerID_LEDGER_ID_L_PUB:
		return s.ls.Pub().CanRead(reader)
	case ledgerv1.LedgerID_LEDGER_ID_L_PRV:
		l := s.ls.Prv(c.GetLedgerChannelId())
		if l == nil {
			return false
		}
		return l.CanRead(reader)
	case ledgerv1.LedgerID_LEDGER_ID_L_COG_IND:
		return s.ls.CogInd().CanRead(reader)
	case ledgerv1.LedgerID_LEDGER_ID_L_COG_ELI:
		return s.ls.CogEli().CanRead(reader)
	case ledgerv1.LedgerID_LEDGER_ID_L_CTRL:
		return s.ls.Ctrl().CanRead(reader)
	default:
		return false
	}
}

// ViewRequestSigningBytes returns the canonical bytes a reader must sign
// to authenticate a ViewRequest: reader_pubkey || from_tick (big-endian).
// Exported because gRPC clients (and the validation experiments) need to
// produce the exact same bytes.
func ViewRequestSigningBytes(readerPubkey []byte, fromTick uint64) []byte {
	buf := make([]byte, 0, len(readerPubkey)+8)
	buf = append(buf, readerPubkey...)
	var tickBuf [8]byte
	binary.BigEndian.PutUint64(tickBuf[:], fromTick)
	return append(buf, tickBuf[:]...)
}

func viewRequestSigningBytes(req *kernelv1.ViewRequest) []byte {
	return ViewRequestSigningBytes(req.GetReaderPubkey(), req.GetFromTick())
}
