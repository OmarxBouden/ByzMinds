// byzminds-test-kernel-1agent is a tiny helper used by the agent/
// pytest smoke test (test_kernel_smoke.py). It boots a kernel with:
//
//   - A one-phase, three-round scenario (so the scheduler dispatches
//     three views).
//   - One spawned agent named "reviewer_01" with the agent_pubkey
//     supplied via --agent-pubkey-hex.
//   - The gRPC servers (Kernel + Handler) on --addr.
//
// The scheduler runs in the background, advancing freely. The Python
// agent connects via Subscribe + SubmitEvent and the bridge wiring
// (api/subscribe.go + scheduler) carries the committed event back as
// a CommitReceipt.
//
// Used only by the test harness; not part of the production CLI.
package main

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"flag"
	"log"
	"math/rand"
	"net"
	"time"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:0", "listen address")
	agentPubHex := flag.String("agent-pubkey-hex", "", "agent Ed25519 public key (32 B hex)")
	rounds := flag.Uint("rounds", 3, "scenario rounds (== ticks dispatched)")
	kElicit := flag.Uint("k-elicit", 0, "scheduler K_elicit; 0 disables elicit pass (Step 4 default)")
	flag.Parse()

	if *agentPubHex == "" {
		log.Fatal("--agent-pubkey-hex required")
	}
	agentPub, err := hex.DecodeString(*agentPubHex)
	if err != nil || len(agentPub) != 32 {
		log.Fatalf("--agent-pubkey-hex must be 32-byte hex, got %d", len(agentPub))
	}

	rng := rand.New(rand.NewSource(42))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)
	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		log.Fatalf("ledger.New: %v", err)
	}
	h := handler.New(ls)
	sch := scheduler.New(ls, h, 30*time.Second)
	sch.SetKElicit(uint32(*kElicit))

	if err := h.LoadScenario(&handler.ScenarioState{
		Name:         "smoke_1agent",
		YAMLHash:     "smoke",
		TaskArtifact: "n/a",
		Phases: []handler.PhaseSpec{
			{Name: "deliberation", Rounds: uint32(*rounds), AvailableTools: []string{"yield"}},
		},
	}); err != nil {
		log.Fatalf("LoadScenario: %v", err)
	}
	if _, err := h.SpawnAgent(&handlerv1.SpawnAgentRequest{
		AgentId:     "reviewer_01",
		AgentPubkey: agentPub,
		Role:        "reviewer",
	}); err != nil {
		log.Fatalf("SpawnAgent: %v", err)
	}

	srv := api.New(ls)
	srv.AttachStep2(h, sch)
	hsrv := api.NewHandlerServer(h)

	grpcServer := grpc.NewServer()
	srv.Register(grpcServer)
	hsrv.Register(grpcServer)

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	log.Printf("byzminds-test-kernel-1agent: listening on %s", lis.Addr())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		// Block until the Python agent has called Subscribe and the
		// bridge is attached. Otherwise the scheduler would dispatch
		// synthetic Yields before the agent has a chance to see any
		// views.
		waitCtx, waitCancel := context.WithTimeout(ctx, 30*time.Second)
		defer waitCancel()
		if err := sch.WaitForAllLiveAgentsAttached(waitCtx); err != nil {
			log.Printf("wait-for-attach: %v", err)
			return
		}
		log.Printf("byzminds-test-kernel-1agent: all agents attached; advancing scheduler")
		if err := sch.RunUntil(ctx, uint64(*rounds)); err != nil {
			log.Printf("scheduler stopped: %v", err)
		}
	}()
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Serve: %v", err)
	}
}
