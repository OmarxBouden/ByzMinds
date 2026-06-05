//go:build ignore

// Experiment 004 — tick determinism.
//
// Plan reference: byzminds-step2-implementation.md §"Validation
// experiments / 004".
//
// Question. Do N stub agents under a fixed scenario produce a
// bit-identical manifest across 100 runs?
//
// Setup. delegated_review_stub.yaml (5 stubs: 2 echo, 2 mirror, 1 silent;
// 4 rounds across two phases). Same kernel seed, same stub keys
// (deterministic from the scenario's pubkey_seed), no goroutines per
// agent — the scheduler drives stubs synchronously in-process.
//
// Expected. All 100 runs produce identical chain_hash. The only
// nondeterminism sources are Go map iteration (we sort all collections
// before commit), time.Now() (we pin commit_unix_nanos and exclude it
// from the chain hash), and goroutine scheduling (no per-agent
// goroutines under the in-process transport — the scheduler calls each
// stub serially in a deterministic order).
//
// Decision criterion. 100/100 identical chain_hash → pass.
//
// Run with: cd kernel && go run experiments/004_tick_determinism.go

package main

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scenario"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	"github.com/byzminds/byzminds/kernel/internal/stubs"
)

const (
	nRuns        = 100
	kernelSeed   = 42
	scenarioPath = "../scenarios/delegated_review_stub.yaml"
)

func main() {
	abs, _ := filepath.Abs(scenarioPath)
	lr, err := scenario.LoadFile(abs)
	if err != nil {
		fail("load scenario: %v", err)
	}
	var refHash []byte
	matches := 0
	for run := 0; run < nRuns; run++ {
		head, err := runOnce(lr)
		if err != nil {
			fail("run %d: %v", run, err)
		}
		if run == 0 {
			refHash = head
			matches++
			continue
		}
		if bytesEqual(head, refHash) {
			matches++
		} else {
			fmt.Fprintf(os.Stderr, "DRIFT at run %d: %x vs ref %x\n", run, head, refHash)
		}
	}
	fmt.Printf("experiment=004 scenario=%s ticks=%d agents=%d runs=%d matching_chain_heads=%d/%d ref_chain_hash=%s\n",
		lr.Spec.Name, lr.TotalTicks, len(lr.Spec.Agents), nRuns, matches, nRuns, hex.EncodeToString(refHash))
	if matches != nRuns {
		fmt.Println("FAIL: not all runs produced the same chain_hash")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func runOnce(lr *scenario.LoadResult) ([]byte, error) {
	rng := rand.New(rand.NewSource(kernelSeed))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)
	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		return nil, err
	}
	h := handler.New(ls)
	sch := scheduler.New(ls, h, time.Second)
	if err := lr.Apply(h, sch); err != nil {
		return nil, err
	}
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		stub := stubs.New(a.ID, stubs.Lookup(a.StubPolicy), kp.Pubkey, kp.Privkey, ls)
		if err := sch.AttachAgent(a.ID, stub); err != nil {
			return nil, err
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := sch.RunUntil(ctx, lr.TotalTicks); err != nil {
		return nil, err
	}
	return ls.ChainHead(), nil
}

func bytesEqual(a, b []byte) bool {
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
	fmt.Fprintf(os.Stderr, "experiment 004: "+format+"\n", args...)
	os.Exit(1)
}
