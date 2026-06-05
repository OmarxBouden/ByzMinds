//go:build ignore

// Experiment 003 — replay determinism.
//
// Plan reference: byzminds-step1-plan.md §"Validation experiment for Step 1".
//
// Question. Does manifest replay reproduce the run bit-for-bit?
//
// Setup. Run a 1000-event scenario through the kernel (one emitter,
// seqs 1..1000 to L_pub). Write the manifest. Replay it 1000 times
// against a fresh kernel; for each replay, compare the manifest's
// final_chain_hash and every event's chain_hash byte-for-byte.
//
// Decision criterion. 1000/1000 successful replays with matching
// chain_hashes and matching per-event canonical bytes → pass.
//
// Run with:  cd kernel && go run experiments/003_replay_determinism.go

package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/manifest"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

func main() {
	nEvents := flag.Int("events", 1000, "events in the scenario")
	nReplays := flag.Int("replays", 1000, "number of replays to perform")
	flag.Parse()

	// 1. Run the original scenario.
	rng := rand.New(rand.NewSource(0xb172_5e7))
	rPub, _, _ := ed25519.GenerateKey(rng)
	_, kPriv, _ := ed25519.GenerateKey(rng)
	emPub, emPriv, _ := ed25519.GenerateKey(rand.New(rand.NewSource(0xe177_e2)))

	original, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		fail("ledger.New (original): %v", err)
	}
	srv := api.New(original)
	for i := 1; i <= *nEvents; i++ {
		payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: "x"})
		env := &eventsv1.EventEnvelope{
			EmitterPubkey:     emPub,
			Tick:              1,
			SequencePerLedger: uint64(i),
			EventType:         "Speak",
			Payload:           payload,
		}
		sig, _ := crypto.SignEnvelope(emPriv, env)
		env.Signature = sig
		rec, err := srv.SubmitEvent(context.Background(), env)
		if err != nil {
			fail("submit %d: %v", i, err)
		}
		if !rec.GetCommitted() {
			fail("submit %d rejected: %s", i, rec.GetRejectionReason())
		}
	}
	originalHead := original.ChainHead()
	originalLog := original.CommittedLog()

	// 2. Write the manifest to memory.
	var buf bytes.Buffer
	if err := manifest.Write(&buf, manifest.Header{KernelVersion: "exp003"}, original, manifest.InitialState{}); err != nil {
		fail("manifest.Write: %v", err)
	}
	manBytes := buf.Bytes()

	// 3. Replay nReplays times. Each replay verifies head match and per-event byte equality.
	t0 := time.Now()
	successes := 0
	for r := 0; r < *nReplays; r++ {
		man, err := manifest.Read(bytes.NewReader(manBytes))
		if err != nil {
			fail("replay %d: manifest.Read: %v", r, err)
		}
		fresh, err := ledger.New(ledger.Config{
			Researcher: rPub,
			KernelPriv: kPriv,
			CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
		})
		if err != nil {
			fail("replay %d: ledger.New: %v", r, err)
		}
		head, err := manifest.Replay(man, fresh)
		if err != nil {
			fail("replay %d: %v", r, err)
		}
		if !bytes.Equal(head, originalHead) {
			fail("replay %d: chain head drift", r)
		}
		// Byte-for-byte check on every committed event.
		got := fresh.CommittedLog()
		if len(got) != len(originalLog) {
			fail("replay %d: event count drift (%d vs %d)", r, len(got), len(originalLog))
		}
		for i := range got {
			a, _ := proto.MarshalOptions{Deterministic: true}.Marshal(got[i])
			b, _ := proto.MarshalOptions{Deterministic: true}.Marshal(originalLog[i])
			if !bytes.Equal(a, b) {
				fail("replay %d event %d: byte drift\n  orig=%x\n  got =%x", r, i, sha256.Sum256(b), sha256.Sum256(a))
			}
		}
		successes++
	}
	elapsed := time.Since(t0)

	fmt.Printf("experiment=003 events=%d replays=%d successes=%d/%d elapsed_seconds=%.2f replay_rate_per_sec=%.1f original_chain_hash=%s\n",
		*nEvents, *nReplays, successes, *nReplays, elapsed.Seconds(), float64(*nReplays)/elapsed.Seconds(), hex.EncodeToString(originalHead))
	if successes != *nReplays {
		fmt.Println("FAIL: not all replays succeeded")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "experiment 003: "+format+"\n", args...)
	os.Exit(1)
}
