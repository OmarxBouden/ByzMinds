//go:build ignore

// Experiment 007 — channel lifecycle.
//
// Plan reference: byzminds-step2-implementation.md §"Validation
// experiments / 007".
//
// Question. When an agent emits OpenChannelReq, does the handler
// auto-approve and mint a new L_prv ledger that's writable by members
// on the next tick? Is the new channel correctly hidden from non-members?
//
// Setup. 3 agents (reviewer_01..03). Custom in-process transports:
//   - reviewer_01: emit OpenChannelReq([reviewer_01, reviewer_03]) at
//     tick 2; speak on the new private channel from tick 3 onward.
//   - reviewer_02: always speak on public (non-member of new channel).
//   - reviewer_03: always speak on public (member but quiet on private).
//
// Expected:
//   1. L_ctrl contains OpenChannelReq committed at tick 2.
//   2. L_ctrl contains Handler_OpenChannel committed at tick 3.
//   3. L_prv:ch_auto_1 (or whatever kernel allocated) holds reviewer_01's
//      Speak from tick 3.
//   4. reviewer_02's View at tick 3 does NOT contain the private channel
//      in channel_memberships nor channel_histories.
//
// Decision criterion. All four assertions hold → pass.
//
// Run with: cd kernel && go run experiments/007_channel_lifecycle.go

package main

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

const (
	totalTicks    = 5
	openReqTick   = 2
)

type r01Transport struct {
	pub  ed25519.PublicKey
	priv ed25519.PrivateKey
	ls   *ledger.LedgerSet
}

func (t *r01Transport) Tick(_ context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	switch v.GetTick() {
	case openReqTick:
		// Emit OpenChannelReq via an envelope routed by schema to L_ctrl.
		req := &eventsv1.OpenChannelReq{ProposedMembers: []string{"reviewer_01", "reviewer_03"}}
		payload, _ := crypto.CanonicalBytes(req)
		dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}
		env := &eventsv1.EventEnvelope{
			EmitterPubkey:     t.pub,
			Tick:              v.GetTick(),
			SequencePerLedger: t.ls.NextSeqFor(t.pub, dest),
			EventType:         "OpenChannelReq",
			Payload:           payload,
		}
		sig, _ := crypto.SignEnvelope(t.priv, env)
		env.Signature = sig
		return env, nil
	default:
		// Speak on the private channel if present, else public.
		channel := "public"
		for _, c := range v.GetChannelMemberships() {
			if c != "public" {
				channel = c
				break
			}
		}
		dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}
		if channel != "public" {
			dest = ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: channel}
		}
		payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: channel, Content: fmt.Sprintf("r01 tick=%d on=%s", v.GetTick(), channel)})
		env := &eventsv1.EventEnvelope{
			EmitterPubkey:     t.pub,
			Tick:              v.GetTick(),
			SequencePerLedger: t.ls.NextSeqFor(t.pub, dest),
			EventType:         "Speak",
			Payload:           payload,
		}
		sig, _ := crypto.SignEnvelope(t.priv, env)
		env.Signature = sig
		return env, nil
	}
}

type publicSpeaker struct {
	id   string
	pub  ed25519.PublicKey
	priv ed25519.PrivateKey
	ls   *ledger.LedgerSet
}

func (t *publicSpeaker) Tick(_ context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	dest := ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}
	payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: fmt.Sprintf("%s tick=%d", t.id, v.GetTick())})
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     t.pub,
		Tick:              v.GetTick(),
		SequencePerLedger: t.ls.NextSeqFor(t.pub, dest),
		EventType:         "Speak",
		Payload:           payload,
	}
	sig, _ := crypto.SignEnvelope(t.priv, env)
	env.Signature = sig
	return env, nil
}

// captureViews wraps an existing transport, capturing the most recent
// view per tick for assertion later.
type captureViews struct {
	inner  scheduler.AgentTransport
	views  *[]*viewv1.View
	mu     *sync.Mutex
}

func (c *captureViews) Tick(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	c.mu.Lock()
	*c.views = append(*c.views, v)
	c.mu.Unlock()
	return c.inner.Tick(ctx, v)
}

func main() {
	rng := rand.New(rand.NewSource(42))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)

	pubs := make(map[string]ed25519.PublicKey)
	privs := make(map[string]ed25519.PrivateKey)
	for i, id := range []string{"reviewer_01", "reviewer_02", "reviewer_03"} {
		krng := rand.New(rand.NewSource(int64(100 + i)))
		pubs[id], privs[id], _ = ed25519.GenerateKey(krng)
	}

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
	if err := h.LoadScenario(&handler.ScenarioState{
		Name:         "channel_lifecycle_exp007",
		YAMLHash:     "exp007",
		TaskArtifact: "n/a",
		Phases: []handler.PhaseSpec{
			{Name: "deliberation", Rounds: totalTicks, AvailableTools: []string{"speak", "yield"}},
		},
	}); err != nil {
		fail("LoadScenario: %v", err)
	}
	for _, id := range []string{"reviewer_01", "reviewer_02", "reviewer_03"} {
		if _, err := h.SpawnAgent(&handlerv1.SpawnAgentRequest{
			AgentId: id, AgentPubkey: pubs[id], Role: "reviewer",
		}); err != nil {
			fail("Spawn %s: %v", id, err)
		}
	}

	r01 := &r01Transport{pub: pubs["reviewer_01"], priv: privs["reviewer_01"], ls: ls}
	r02 := &publicSpeaker{id: "reviewer_02", pub: pubs["reviewer_02"], priv: privs["reviewer_02"], ls: ls}
	r03 := &publicSpeaker{id: "reviewer_03", pub: pubs["reviewer_03"], priv: privs["reviewer_03"], ls: ls}

	var mu sync.Mutex
	r02Views := &[]*viewv1.View{}

	_ = sch.AttachAgent("reviewer_01", r01)
	_ = sch.AttachAgent("reviewer_02", &captureViews{inner: r02, views: r02Views, mu: &mu})
	_ = sch.AttachAgent("reviewer_03", r03)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := sch.RunUntil(ctx, totalTicks); err != nil {
		fail("RunUntil: %v", err)
	}

	// --- 1. L_ctrl contains OpenChannelReq at tick 2.
	ctrlEvents := ls.Ctrl().Snapshot()
	openReqFound := false
	openApprovedFound := false
	var approvedTick uint64
	for _, c := range ctrlEvents {
		t := c.GetEnvelope().GetEventType()
		switch t {
		case "OpenChannelReq":
			if c.GetEnvelope().GetTick() == openReqTick {
				openReqFound = true
			}
		case "Handler_OpenChannel":
			openApprovedFound = true
			approvedTick = c.GetEnvelope().GetTick()
		}
	}
	if !openReqFound {
		fail("no OpenChannelReq at tick %d in L_ctrl", openReqTick)
	}
	if !openApprovedFound {
		fail("no Handler_OpenChannel in L_ctrl")
	}
	if approvedTick != openReqTick+1 {
		fail("OpenChannel approved at tick %d, want %d", approvedTick, openReqTick+1)
	}

	// --- 2. New ledger exists; reviewer_01 wrote to it at tick 3+.
	var prvName string
	for _, name := range []string{"ch_auto_1", "ch_auto_2"} {
		if l := ls.Prv(name); l != nil {
			prvName = name
			break
		}
	}
	if prvName == "" {
		fail("no auto-allocated private channel found")
	}
	prvEvents := ls.Prv(prvName).Snapshot()
	if len(prvEvents) == 0 {
		fail("private channel %s is empty; reviewer_01 should have spoken there at tick %d+", prvName, openReqTick+1)
	}
	if prvEvents[0].GetEnvelope().GetTick() < openReqTick+1 {
		fail("first private write at tick %d, want >= %d", prvEvents[0].GetEnvelope().GetTick(), openReqTick+1)
	}

	// --- 3. reviewer_02's views never include the private channel.
	leaked := false
	for _, v := range *r02Views {
		for _, c := range v.GetChannelMemberships() {
			if c == prvName {
				leaked = true
			}
		}
		for _, ch := range v.GetChannelHistories() {
			if ch.GetChannelId() == prvName {
				leaked = true
			}
		}
	}

	fmt.Printf("experiment=007 total_ticks=%d open_req_tick=%d approved_tick=%d private_channel=%s private_events=%d r02_leak=%v\n",
		totalTicks, openReqTick, approvedTick, prvName, len(prvEvents), leaked)
	if leaked {
		fmt.Println("FAIL: reviewer_02 saw the private channel in their view")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "experiment 007: "+format+"\n", args...)
	os.Exit(1)
}
