// byzminds-run loads a scenario YAML, runs it end-to-end against an
// in-process kernel + in-process stub agents (no gRPC), and writes a
// replayable manifest to disk.
//
// Usage:
//   byzminds-run scenarios/delegated_review_stub.yaml --seed 42 --manifest /tmp/run.json.gz
package main

import (
	"context"
	"crypto/ed25519"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/manifest"
	"github.com/byzminds/byzminds/kernel/internal/scenario"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	"github.com/byzminds/byzminds/kernel/internal/stubs"
)

var version = "dev"

func main() {
	seed := flag.Int64("seed", 1, "kernel + researcher keygen seed")
	manifestPath := flag.String("manifest", "", "output manifest path (gzipped JSON); empty ⇒ no write")
	tickTimeout := flag.Duration("tick-timeout", 30*time.Second, "per-tick agent emission budget")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: byzminds-run <scenario.yaml>")
		os.Exit(2)
	}
	yamlPath := flag.Arg(0)

	lr, err := scenario.LoadFile(yamlPath)
	if err != nil {
		log.Fatalf("load scenario: %v", err)
	}
	log.Printf("byzminds-run %s: loaded scenario %q (yaml_hash=%s total_ticks=%d agents=%d)",
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
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		stub := stubs.New(a.ID, stubs.Lookup(a.StubPolicy), kp.Pubkey, kp.Privkey, ls)
		if err := sch.AttachAgent(a.ID, stub); err != nil {
			log.Fatalf("attach %s: %v", a.ID, err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := sch.RunUntil(ctx, lr.TotalTicks); err != nil {
		log.Fatalf("RunUntil: %v", err)
	}

	log.Printf("scenario complete: ticks=%d chain_head=%x committed_events=%d",
		sch.CurrentTick(), ls.ChainHead(), len(ls.CommittedLog()))

	if *manifestPath != "" {
		f, err := os.Create(*manifestPath)
		if err != nil {
			log.Fatalf("create manifest: %v", err)
		}
		defer f.Close()
		if err := manifest.Write(f, manifest.Header{KernelVersion: version}, ls, manifest.InitialState{}); err != nil {
			log.Fatalf("write manifest: %v", err)
		}
		log.Printf("manifest written to %s", *manifestPath)
	}
}
