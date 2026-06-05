// byzminds-panel runs a scenario end-to-end against the gRPC kernel with
// EXTERNAL agents (Python byzminds-agent processes) attaching via Subscribe,
// then writes a replayable manifest. It is byzminds-run's scenario+scheduler
// path wired onto the gRPC server (api.AttachStep2 + the subscriber bridge),
// instead of in-process Go stubs.
//
// Flow:
//   1. Load scenario, build ledger/handler/scheduler, lr.Apply (spawns agents).
//   2. Export each agent's keypair (Go 64-byte seed||pub) to --keys-dir so the
//      Python agents can sign.
//   3. Serve gRPC; agents Subscribe -> the bridge AttachAgent's them.
//   4. WaitForAllLiveAgentsAttached, RunUntil(TotalTicks), write manifest, exit.
//
// Usage:
//   byzminds-panel scenarios/M5/run.yaml --addr 127.0.0.1:7777 \
//       --keys-dir /tmp/m5keys --manifest /tmp/run.json.gz --seed 42
package main

import (
	"context"
	"crypto/ed25519"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/manifest"
	"github.com/byzminds/byzminds/kernel/internal/scenario"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
)

var version = "dev"

func main() {
	addr := flag.String("addr", "127.0.0.1:7777", "gRPC listen address")
	seed := flag.Int64("seed", 1, "kernel + researcher keygen seed")
	manifestPath := flag.String("manifest", "", "output manifest path (gzipped JSON)")
	keysDir := flag.String("keys-dir", "", "directory to write per-agent keypair files (<id>.key, 64-byte seed||pub)")
	tickTimeout := flag.Duration("tick-timeout", 120*time.Second, "per-tick agent emission budget (LLM latency)")
	attachTimeout := flag.Duration("attach-timeout", 120*time.Second, "how long to wait for all agents to Subscribe")
	agentTheta := flag.String("agent-theta", "", "per-agent induced disposition recorded on L_cog_ind, "+
		"as \"id=dial:strength,...\" (e.g. reviewer_01=collude:strong). Maps to a length-6 theta vector in "+
		"DIALS order; agents not listed get a zero vector (honest).")
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatalf("usage: byzminds-panel <scenario.yaml> --keys-dir DIR --manifest PATH")
	}

	lr, err := scenario.LoadFile(flag.Arg(0))
	if err != nil {
		log.Fatalf("load scenario: %v", err)
	}
	if *agentTheta != "" {
		thetaByAgent, err := parseAgentTheta(*agentTheta)
		if err != nil {
			log.Fatalf("--agent-theta: %v", err)
		}
		for i := range lr.Spec.Agents {
			if th, ok := thetaByAgent[lr.Spec.Agents[i].ID]; ok {
				lr.Spec.Agents[i].Theta = th
			}
		}
		log.Printf("agent-theta: set induced disposition for %d agent(s)", len(thetaByAgent))
	}
	log.Printf("byzminds-panel %s: scenario %q yaml_hash=%s total_ticks=%d agents=%d",
		version, lr.Spec.Name, lr.YAMLHash, lr.TotalTicks, len(lr.Spec.Agents))

	rng := rand.New(rand.NewSource(*seed))
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
	sch := scheduler.New(ls, h, *tickTimeout)
	if err := lr.Apply(h, sch); err != nil {
		log.Fatalf("scenario.Apply: %v", err)
	}

	// Export agent keypairs so the external Python agents can sign.
	if *keysDir == "" {
		log.Fatalf("--keys-dir is required (external agents need their keypairs)")
	}
	if err := os.MkdirAll(*keysDir, 0o700); err != nil {
		log.Fatalf("mkdir keys-dir: %v", err)
	}
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		if err := os.WriteFile(filepath.Join(*keysDir, a.ID+".key"), []byte(kp.Privkey), 0o600); err != nil {
			log.Fatalf("write keypair %s: %v", a.ID, err)
		}
	}
	log.Printf("exported %d agent keypairs to %s", len(lr.Spec.Agents), *keysDir)

	// gRPC server with the Step-2 bridge attached.
	kernelSrv := api.New(ls)
	kernelSrv.AttachStep2(h, sch)
	grpcServer := grpc.NewServer()
	kernelSrv.Register(grpcServer)
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *addr, err)
	}
	go func() {
		if err := grpcServer.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			log.Printf("grpc serve stopped: %v", err)
		}
	}()
	log.Printf("byzminds-panel listening on %s; waiting for %d agents to subscribe", *addr, len(lr.Spec.Agents))

	// Wait for all live agents to attach (Subscribe), then drive the scenario.
	attachCtx, cancelAttach := context.WithTimeout(context.Background(), *attachTimeout)
	if err := sch.WaitForAllLiveAgentsAttached(attachCtx); err != nil {
		cancelAttach()
		log.Fatalf("waiting for agents to subscribe: %v", err)
	}
	cancelAttach()
	log.Printf("all agents attached; running %d ticks", lr.TotalTicks)

	runCtx, cancelRun := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancelRun()
	if err := sch.RunUntil(runCtx, lr.TotalTicks); err != nil {
		log.Fatalf("RunUntil: %v", err)
	}
	log.Printf("scenario complete: ticks=%d chain_head=%x committed_events=%d",
		sch.CurrentTick(), ls.ChainHead(), len(ls.CommittedLog()))

	if *manifestPath != "" {
		f, err := os.Create(*manifestPath)
		if err != nil {
			log.Fatalf("create manifest: %v", err)
		}
		if err := manifest.Write(f, manifest.Header{KernelVersion: version}, ls, manifest.InitialState{}); err != nil {
			f.Close()
			log.Fatalf("write manifest: %v", err)
		}
		f.Close()
		log.Printf("manifest written to %s", *manifestPath)
	}
	// Forceful stop: agents hold long-lived Subscribe streams, so
	// GracefulStop would block on them. The manifest is already written,
	// so we stop immediately; agents see their stream close and exit.
	grpcServer.Stop()
}

// dialOrder is the canonical length-6 theta ordering. It MUST mirror Python
// byzminds_agent.DIALS (asserted by TestDialOrderMatchesPython).
var dialOrder = []string{"authority", "bandwagon", "sycophancy", "free_ride", "collude", "deceive"}

var strengthMag = map[string]float64{"none": 0, "mild": 1.0 / 3.0, "moderate": 2.0 / 3.0, "strong": 1.0}

// thetaForPersona maps (dial, strength) to a length-6 theta vector in dialOrder:
// zero everywhere except the dial's index, set to the strength magnitude.
func thetaForPersona(dial, strength string) ([]float64, error) {
	theta := make([]float64, len(dialOrder))
	if dial == "" || strength == "none" {
		return theta, nil
	}
	mag, ok := strengthMag[strength]
	if !ok {
		return nil, fmt.Errorf("unknown strength %q", strength)
	}
	idx := -1
	for i, d := range dialOrder {
		if d == dial {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil, fmt.Errorf("unknown dial %q", dial)
	}
	theta[idx] = mag
	return theta, nil
}

// parseAgentTheta parses "id=dial:strength,id2=dial2:strength2" into a map of
// agent id -> theta vector.
func parseAgentTheta(spec string) (map[string][]float64, error) {
	out := make(map[string][]float64)
	for _, entry := range strings.Split(spec, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		id, persona, ok := strings.Cut(entry, "=")
		if !ok {
			return nil, fmt.Errorf("bad entry %q (want id=dial:strength)", entry)
		}
		dial, strength, ok := strings.Cut(persona, ":")
		if !ok {
			return nil, fmt.Errorf("bad persona %q in %q (want dial:strength)", persona, entry)
		}
		theta, err := thetaForPersona(strings.TrimSpace(dial), strings.TrimSpace(strength))
		if err != nil {
			return nil, err
		}
		out[strings.TrimSpace(id)] = theta
	}
	return out, nil
}
