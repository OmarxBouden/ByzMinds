//go:build ignore

// Experiment 002 — signature verification throughput.
//
// Plan reference: byzminds-step1-plan.md §"Validation experiment for Step 1".
//
// Question. Can the kernel verify and commit enough signed envelopes
// per second to handle realistic Stage A workloads (≥ 1k events/sec)?
//
// Setup. Pre-generate one emitter and 100,000 signed envelopes with
// monotonically increasing sequence_per_ledger 1..100000 to L_pub.
// Pre-signing happens off the clock; then the timed phase calls
// SubmitEvent serially against an in-process api.Server (no gRPC wire).
// SubmitEvent runs the production verify → schema-validate → commit
// pipeline.
//
// Decision criterion. ≥ 1k events/sec on the developer laptop → pass.
//
// Run with:  cd kernel && go run experiments/002_signature_throughput.go

package main

import (
	"context"
	"crypto/ed25519"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

const minRate = 1000.0 // events/sec floor for pass

func main() {
	n := flag.Int("n", 100_000, "number of pre-signed events to verify+commit")
	flag.Parse()

	rng := rand.New(rand.NewSource(0xb172_5e7))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)
	emPub, emPriv, err := ed25519.GenerateKey(rand.New(rand.NewSource(0xe177_e2)))
	if err != nil {
		fail("emitter keygen: %v", err)
	}

	envs := make([]*eventsv1.EventEnvelope, *n)
	t0 := time.Now()
	for i := 0; i < *n; i++ {
		payload, err := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: "x"})
		if err != nil {
			fail("payload %d: %v", i, err)
		}
		env := &eventsv1.EventEnvelope{
			EmitterPubkey:     emPub,
			Tick:              1,
			SequencePerLedger: uint64(i + 1),
			EventType:         "Speak",
			Payload:           payload,
		}
		sig, err := crypto.SignEnvelope(emPriv, env)
		if err != nil {
			fail("sign %d: %v", i, err)
		}
		env.Signature = sig
		envs[i] = env
	}
	signTime := time.Since(t0)

	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		fail("ledger.New: %v", err)
	}
	srv := api.New(ls)

	t1 := time.Now()
	for i, env := range envs {
		rec, err := srv.SubmitEvent(context.Background(), env)
		if err != nil {
			fail("submit %d: %v", i, err)
		}
		if !rec.GetCommitted() {
			fail("submit %d rejected: %s", i, rec.GetRejectionReason())
		}
	}
	verifyCommitTime := time.Since(t1)

	rate := float64(*n) / verifyCommitTime.Seconds()
	signRate := float64(*n) / signTime.Seconds()
	fmt.Printf("experiment=002 events=%d sign_seconds=%.3f sign_rate_eps=%.0f verify_commit_seconds=%.3f verify_commit_rate_eps=%.0f min_rate_eps=%.0f\n",
		*n, signTime.Seconds(), signRate, verifyCommitTime.Seconds(), rate, minRate)
	if rate < minRate {
		fmt.Printf("FAIL: verify+commit rate %.0f eps below floor of %.0f eps\n", rate, minRate)
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "experiment 002: "+format+"\n", args...)
	os.Exit(1)
}
