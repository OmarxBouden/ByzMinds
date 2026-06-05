package integration

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/manifest"
	"github.com/byzminds/byzminds/kernel/internal/scenario"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
	"github.com/byzminds/byzminds/kernel/internal/stubs"
)

// TestEndToEndStubScenario runs delegated_review_stub.yaml end to end:
// load the YAML, spawn agents + stubs, run all ticks, write a manifest,
// replay against a fresh kernel, assert chain head matches.
//
// This is the Step 2 sibling of Step 1's Experiment 003: a real-world
// integration that exercises scheduler + handler + view + stubs + L_cog_ind.
func TestEndToEndStubScenario(t *testing.T) {
	ls, _, _ := newKernel(t)
	h := handler.New(ls)
	sch := scheduler.New(ls, h, time.Second)

	yamlPath, err := filepath.Abs("../../../scenarios/delegated_review_stub.yaml")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	lr, err := scenario.LoadFile(yamlPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if err := lr.Apply(h, sch); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	// Attach stubs.
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		stub := stubs.New(a.ID, stubs.Lookup(a.StubPolicy), kp.Pubkey, kp.Privkey, ls)
		if err := sch.AttachAgent(a.ID, stub); err != nil {
			t.Fatalf("AttachAgent %s: %v", a.ID, err)
		}
	}

	// Run all scenario ticks.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := sch.RunUntil(ctx, lr.TotalTicks); err != nil {
		t.Fatalf("RunUntil: %v", err)
	}

	// Write + replay the manifest.
	var buf bytes.Buffer
	if err := manifest.Write(&buf, manifest.Header{KernelVersion: "step2-it"}, ls, manifest.InitialState{}); err != nil {
		t.Fatalf("manifest.Write: %v", err)
	}
	man, err := manifest.Read(&buf)
	if err != nil {
		t.Fatalf("manifest.Read: %v", err)
	}
	fresh, _, _ := newKernel(t)
	got, err := manifest.Replay(man, fresh)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if !bytes.Equal(got, ls.ChainHead()) {
		t.Fatalf("replay chain head drift")
	}
}

func newKernel(t *testing.T) (*ledger.LedgerSet, []byte, []byte) {
	t.Helper()
	rPub, _, _ := crypto.GenerateKey()
	_, kPriv, _ := crypto.GenerateKey()
	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		t.Fatalf("ledger.New: %v", err)
	}
	return ls, rPub, kPriv
}
