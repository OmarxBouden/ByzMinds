// byzminds-stub-agent is the minimal cross-process gRPC stub: it dials
// the kernel, subscribes via Kernel.Subscribe with its agent_id's
// signed signature, and replies to every view with a Yield envelope.
//
// Step 2's regression-grade scenario harness (byzminds-run + Experiments
// 004–007) uses *in-process* stubs from kernel/internal/stubs for
// determinism and speed. This binary exists as a cross-process smoke
// test for the gRPC Subscribe / SubmitEvent surface — once Step 3 lands
// the Python adapter, the same surface must keep working. To keep this
// binary tractable in Step 2, the policy is hard-coded to "always
// yield" so the agent only ever emits one event_type at one per-emitter
// seq counter (no kernel-synthesized races on the agent's pubkey).
//
// Usage:
//   byzminds-stub-agent --kernel 127.0.0.1:7777 \
//     --agent-id reviewer_01 \
//     --agent-priv-hex <128 hex chars>
package main

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"errors"
	"flag"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	kernelv1 "github.com/byzminds/byzminds/proto/kernelv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

func main() {
	addr := flag.String("kernel", "127.0.0.1:7777", "kernel gRPC address")
	agentID := flag.String("agent-id", "", "agent id (must match a SpawnAgent call)")
	privHex := flag.String("agent-priv-hex", "", "hex-encoded agent Ed25519 private key (64 bytes)")
	fromTick := flag.Uint64("from-tick", 0, "starting tick (inclusive)")
	flag.Parse()

	if *agentID == "" {
		log.Fatal("--agent-id required")
	}
	privBytes, err := hex.DecodeString(*privHex)
	if err != nil || len(privBytes) != crypto.PrivateKeySize {
		log.Fatalf("--agent-priv-hex must be %d-byte hex", crypto.PrivateKeySize)
	}
	priv := ed25519.PrivateKey(privBytes)
	pub := priv.Public().(ed25519.PublicKey)

	conn, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("dial %s: %v", *addr, err)
	}
	defer conn.Close()
	cli := kernelv1.NewKernelClient(conn)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		cancel()
	}()

	signingBytes := api.ViewRequestSigningBytes(pub, *fromTick)
	subReq := &kernelv1.SubscribeRequest{
		AgentPubkey: pub,
		FromTick:    *fromTick,
		Signature:   ed25519.Sign(priv, signingBytes),
	}
	stream, err := cli.Subscribe(ctx, subReq)
	if err != nil {
		log.Fatalf("Subscribe: %v", err)
	}
	log.Printf("byzminds-stub-agent: agent=%s policy=always-yield kernel=%s", *agentID, *addr)

	var localSeq uint64
	for {
		v, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Fatalf("stream.Recv: %v", err)
		}
		localSeq++
		env, err := yieldEnvelope(pub, priv, v.GetTick(), localSeq, "stub_agent_binary")
		if err != nil {
			log.Fatalf("yieldEnvelope: %v", err)
		}
		rec, err := cli.SubmitEvent(ctx, env)
		if err != nil {
			log.Fatalf("SubmitEvent: %v", err)
		}
		if !rec.GetCommitted() {
			log.Printf("rejected at tick %d: %s", v.GetTick(), rec.GetRejectionReason())
		}
	}
}

func yieldEnvelope(pub ed25519.PublicKey, priv ed25519.PrivateKey, tick, seq uint64, reason string) (*eventsv1.EventEnvelope, error) {
	payload, err := crypto.CanonicalBytes(&eventsv1.Yield{Reason: reason})
	if err != nil {
		return nil, err
	}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              tick,
		SequencePerLedger: seq,
		EventType:         "Yield",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(priv, env)
	if err != nil {
		return nil, err
	}
	env.Signature = sig
	return env, nil
}

// Silence unused-import-warning in environments where the viewv1 type
// is only used implicitly via the stream's value.
var _ = (*viewv1.View)(nil)
