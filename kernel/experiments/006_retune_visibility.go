//go:build ignore

// Experiment 006 — retune visibility.
//
// Plan reference: byzminds-step2-implementation.md §"Validation
// experiments / 006".
//
// Question. When the handler calls Retune(agent_id, theta_new) mid-
// scenario, is the call recorded in L_ctrl, and does the next
// L_cog_ind snapshot show the new θ?
//
// Setup. Build a 10-tick scenario inline (no YAML — we want the
// scheduling control). 1 echo stub. Pre-schedule a Retune at tick 5
// with theta=[0,0,0,0.5,0,0]. Inspect L_ctrl + L_cog_ind after the
// run.
//
// Expected.
//   - L_ctrl contains a Handler_Retune record committed at tick 5.
//   - L_cog_ind snapshots for the agent show theta=zero at ticks 0–5,
//     theta=[0,0,0,0.5,0,0] at ticks 6+ (retune effective_tick = 6).
//
// Decision criterion. Exact trace + off-by-one boundary as expected
// → pass.
//
// Run with: cd kernel && go run experiments/006_retune_visibility.go

package main

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"math/rand"
	"os"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	"github.com/byzminds/byzminds/kernel/internal/stubs"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
)

const (
	totalTicks   = 10
	retuneAtTick = 5
	agentID      = "reviewer_01"
)

func main() {
	rng := rand.New(rand.NewSource(42))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)
	stubRng := rand.New(rand.NewSource(101))
	agentPub, agentPriv, _ := ed25519.GenerateKey(stubRng)

	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		fail("ledger.New: %v", err)
	}
	h := handler.New(ls)
	sch := scheduler.New(ls, h, time.Second)

	// Scenario: 1 phase, 10 rounds.
	if err := h.LoadScenario(&handler.ScenarioState{
		Name:         "retune_visibility_exp006",
		YAMLHash:     "exp006",
		TaskArtifact: "n/a",
		Phases: []handler.PhaseSpec{
			{Name: "deliberation", Rounds: totalTicks, AvailableTools: []string{"speak", "yield"}},
		},
	}); err != nil {
		fail("LoadScenario: %v", err)
	}
	if _, err := h.SpawnAgent(&handlerv1.SpawnAgentRequest{
		AgentId:     agentID,
		AgentPubkey: agentPub,
		Role:        "reviewer",
		StubPolicy:  string(stubs.PolicyEcho),
	}); err != nil {
		fail("SpawnAgent: %v", err)
	}
	stub := stubs.New(agentID, stubs.PolicyEcho, agentPub, agentPriv, ls)
	if err := sch.AttachAgent(agentID, stub); err != nil {
		fail("AttachAgent: %v", err)
	}

	newTheta := []float64{0, 0, 0, 0.5, 0, 0}
	sch.ScheduleAt(retuneAtTick, func() error {
		_, err := h.Retune(&handlerv1.RetuneRequest{
			AgentId: agentID,
			Theta:   newTheta,
		})
		return err
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := sch.RunUntil(ctx, totalTicks); err != nil {
		fail("RunUntil: %v", err)
	}

	// --- Inspect L_ctrl for the Retune record.
	ctrlEvents := ls.Ctrl().Snapshot()
	var retuneTick uint64
	retuneFound := false
	for _, c := range ctrlEvents {
		if c.GetEnvelope().GetEventType() != "Handler_Retune" {
			continue
		}
		ce := &eventsv1.HandlerControlEvent{}
		if err := proto.Unmarshal(c.GetEnvelope().GetPayload(), ce); err != nil {
			fail("unmarshal HandlerControlEvent: %v", err)
		}
		retuneFound = true
		retuneTick = c.GetEnvelope().GetTick()
	}
	if !retuneFound {
		fail("no Handler_Retune found in L_ctrl")
	}
	if retuneTick != retuneAtTick {
		fail("Retune recorded at tick %d, want %d", retuneTick, retuneAtTick)
	}

	// --- Inspect L_cog_ind for the per-tick θ trace.
	cogEvents := ls.CogInd().Snapshot()
	type snap struct {
		tick  uint64
		theta []float64
	}
	var trace []snap
	for _, c := range cogEvents {
		if c.GetEnvelope().GetEventType() != "CogIndSnapshot" {
			continue
		}
		s := &eventsv1.CogIndSnapshot{}
		if err := proto.Unmarshal(c.GetEnvelope().GetPayload(), s); err != nil {
			fail("unmarshal CogIndSnapshot: %v", err)
		}
		if s.GetAgentId() != agentID {
			continue
		}
		trace = append(trace, snap{tick: c.GetEnvelope().GetTick(), theta: s.GetTheta()})
	}

	mismatches := 0
	for _, t := range trace {
		want := zeroTheta()
		if t.tick > retuneAtTick {
			want = newTheta
		}
		if !thetaEqual(t.theta, want) {
			mismatches++
			fmt.Fprintf(os.Stderr, "tick=%d theta=%v want=%v\n", t.tick, t.theta, want)
		}
	}
	fmt.Printf("experiment=006 total_ticks=%d retune_tick=%d retune_recorded_at_tick=%d cog_ind_snapshots=%d mismatches=%d\n",
		totalTicks, retuneAtTick, retuneTick, len(trace), mismatches)
	if mismatches != 0 {
		fmt.Println("FAIL: L_cog_ind theta trace does not match expected zero→nonzero boundary at tick 6")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func zeroTheta() []float64 { return []float64{0, 0, 0, 0, 0, 0} }

func thetaEqual(a, b []float64) bool {
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

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "experiment 006: "+format+"\n", args...)
	os.Exit(1)
}
