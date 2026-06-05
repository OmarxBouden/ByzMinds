//go:build ignore

// Experiment 005 — view fairness.
//
// Plan reference: byzminds-step2-implementation.md §"Validation
// experiments / 005".
//
// Question. Do two stub agents in the same scenario see consistent
// L_pub content (same messages, same order) in their views?
//
// Setup. Same scenario as 004. After each tick is dispatched, capture
// every agent's View. Compare every agent's
// View.ChannelHistories["public"] against agent_01's for byte-identical
// messages (the public channel is broadcast — agents differ on what
// other channels they belong to but not on what's on public).
//
// We intercept views by wrapping each stub's transport with a
// capturing decorator that stores the *View it was handed before
// returning the envelope. After scheduler.RunUntil, we compare
// captures.
//
// Decision criterion. For every tick, every agent's public history
// must be byte-identical to agent_01's. Any divergence → fail.
//
// Run with: cd kernel && go run experiments/005_view_fairness.go

package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scenario"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	"github.com/byzminds/byzminds/kernel/internal/stubs"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

type capturedView struct {
	tick  uint64
	bytes []byte // canonical-marshal of v.ChannelHistories["public"]
}

type capturingTransport struct {
	inner    scheduler.AgentTransport
	captures *[]capturedView
	mu       *sync.Mutex
}

func (c *capturingTransport) Tick(ctx context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	// Extract public history, marshal canonically.
	var pub *viewv1.ChannelHistory
	for _, ch := range v.GetChannelHistories() {
		if ch.GetChannelId() == "public" {
			pub = ch
			break
		}
	}
	if pub == nil {
		pub = &viewv1.ChannelHistory{ChannelId: "public"}
	}
	buf, err := proto.MarshalOptions{Deterministic: true}.Marshal(pub)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	*c.captures = append(*c.captures, capturedView{tick: v.GetTick(), bytes: buf})
	c.mu.Unlock()
	return c.inner.Tick(ctx, v)
}

func main() {
	abs, _ := filepath.Abs("../scenarios/delegated_review_stub.yaml")
	lr, err := scenario.LoadFile(abs)
	if err != nil {
		fail("load: %v", err)
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
		fail("ledger.New: %v", err)
	}
	h := handler.New(ls)
	sch := scheduler.New(ls, h, time.Second)
	if err := lr.Apply(h, sch); err != nil {
		fail("Apply: %v", err)
	}

	captures := make(map[string]*[]capturedView)
	var muRef sync.Mutex
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		stub := stubs.New(a.ID, stubs.Lookup(a.StubPolicy), kp.Pubkey, kp.Privkey, ls)
		caps := &[]capturedView{}
		captures[a.ID] = caps
		tr := &capturingTransport{inner: stub, captures: caps, mu: &muRef}
		if err := sch.AttachAgent(a.ID, tr); err != nil {
			fail("attach %s: %v", a.ID, err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := sch.RunUntil(ctx, lr.TotalTicks); err != nil {
		fail("run: %v", err)
	}

	// Reference agent — pick the first agent alphabetically.
	var refID string
	for _, a := range lr.Spec.Agents {
		if refID == "" || a.ID < refID {
			refID = a.ID
		}
	}
	ref := captures[refID]
	mismatches := 0
	checked := 0
	for _, a := range lr.Spec.Agents {
		if a.ID == refID {
			continue
		}
		other := captures[a.ID]
		if len(*other) != len(*ref) {
			fail("agent %s has %d captures, ref has %d", a.ID, len(*other), len(*ref))
		}
		for i, c := range *other {
			checked++
			if c.tick != (*ref)[i].tick {
				fail("tick mismatch: %s tick=%d, %s tick=%d at i=%d", a.ID, c.tick, refID, (*ref)[i].tick, i)
			}
			if !bytes.Equal(c.bytes, (*ref)[i].bytes) {
				mismatches++
				fmt.Fprintf(os.Stderr, "DIVERGE tick=%d agent=%s vs ref=%s\n  got=%x\n  ref=%x\n",
					c.tick, a.ID, refID, c.bytes, (*ref)[i].bytes)
			}
		}
	}
	fmt.Printf("experiment=005 scenario=%s ref_agent=%s agents=%d ticks=%d comparisons=%d mismatches=%d\n",
		lr.Spec.Name, refID, len(lr.Spec.Agents), lr.TotalTicks, checked, mismatches)
	if mismatches != 0 {
		fmt.Println("FAIL: public channel histories diverged across agents")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "experiment 005: "+format+"\n", args...)
	os.Exit(1)
}
