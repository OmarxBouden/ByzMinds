//go:build ignore

// Experiment 001 — ordering invariants under concurrent submission.
//
// Plan reference: byzminds-step1-plan.md §"Validation experiment for Step 1".
//
// Question. Given N emitter identities each submitting K signed events
// to L_pub, does the kernel produce a deterministic total order across
// repeated runs?
//
// Setup. nEmitters = 10, eventsPerEmitter = 100. Emitter Ed25519 keys
// are derived from a fixed math/rand seed so the signed envelopes are
// byte-identical across runs. Submission is sequential, in a fixed
// round-robin interleaving (emitter_0 seq 1, emitter_1 seq 1, ...,
// emitter_9 seq 1, emitter_0 seq 2, ...). Per-(emitter, ledger)
// sequencing means no submission ever collides — no rejection, no
// retry, no coordination harness.
//
// Note on "concurrent". The plan describes 10 stub clients submitting
// "as fast as possible". Truly concurrent submission would make the
// commit order goroutine-scheduling-dependent and the chain_hash
// nondeterministic across runs. Per the user's directive, with
// per-(emitter, ledger) sequencing this experiment is run sequentially
// in a fixed interleaving — exercising mixed-emitter ordering through
// the kernel's commit pipeline without needing a coordinating harness.
//
// Decision criterion. 100 fresh-kernel runs over the same 1000-event
// schedule must all produce the same final chain_hash → pass.
//
// Run with:  cd kernel && go run experiments/001_kernel_ordering.go

package main

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

const (
	nEmitters        = 10
	eventsPerEmitter = 100
	nRuns            = 100
	emitterSeedBase  = 0xb172
	kernelKeySeed    = 0xb172_5e7
)

func main() {
	pubs := make([]ed25519.PublicKey, nEmitters)
	privs := make([]ed25519.PrivateKey, nEmitters)
	for i := 0; i < nEmitters; i++ {
		rng := rand.New(rand.NewSource(int64(emitterSeedBase + i)))
		pub, priv, err := ed25519.GenerateKey(rng)
		if err != nil {
			fail("emitter keygen %d: %v", i, err)
		}
		pubs[i] = pub
		privs[i] = priv
	}

	envs := make([]*eventsv1.EventEnvelope, 0, nEmitters*eventsPerEmitter)
	for s := uint64(1); s <= eventsPerEmitter; s++ {
		for i := 0; i < nEmitters; i++ {
			payload, err := crypto.CanonicalBytes(&eventsv1.Speak{
				ChannelId: "public",
				Content:   fmt.Sprintf("e%d-s%d", i, s),
			})
			if err != nil {
				fail("payload e%d s%d: %v", i, s, err)
			}
			env := &eventsv1.EventEnvelope{
				EmitterPubkey:     pubs[i],
				Tick:              1,
				SequencePerLedger: s,
				EventType:         "Speak",
				Payload:           payload,
			}
			sig, err := crypto.SignEnvelope(privs[i], env)
			if err != nil {
				fail("sign e%d s%d: %v", i, s, err)
			}
			env.Signature = sig
			envs = append(envs, env)
		}
	}

	var refHash []byte
	matches := 0
	for run := 0; run < nRuns; run++ {
		ls := newKernelDeterministic()
		srv := api.New(ls)
		for i, env := range envs {
			rec, err := srv.SubmitEvent(context.Background(), env)
			if err != nil {
				fail("run %d env %d submit error: %v", run, i, err)
			}
			if !rec.GetCommitted() {
				fail("run %d env %d rejected: %s", run, i, rec.GetRejectionReason())
			}
		}
		head := ls.ChainHead()
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

	fmt.Printf("experiment=001 emitters=%d events_per_emitter=%d total_events=%d runs=%d matching_chain_heads=%d/%d ref_chain_hash=%s\n",
		nEmitters, eventsPerEmitter, nEmitters*eventsPerEmitter, nRuns, matches, nRuns, hex.EncodeToString(refHash))
	if matches != nRuns {
		fmt.Println("FAIL: not all runs produced the same chain_hash")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func newKernelDeterministic() *ledger.LedgerSet {
	rng := rand.New(rand.NewSource(kernelKeySeed))
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
	return ls
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
	fmt.Fprintf(os.Stderr, "experiment 001: "+format+"\n", args...)
	os.Exit(1)
}
