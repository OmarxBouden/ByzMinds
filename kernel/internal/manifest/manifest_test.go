package manifest

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"testing"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// helpers (mirror ledger_test.go) — kept private here so the manifest
// package has a self-contained test surface.

func fixedTime() ledger.CommitTime { return func() uint64 { return 1_700_000_000_000_000_000 } }

func newSet(t *testing.T, member ed25519.PublicKey) (*ledger.LedgerSet, ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	rPub, _, _ := crypto.GenerateKey()
	kPub, kPriv, _ := crypto.GenerateKey()
	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		PrivateChannels: []ledger.PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []ledger.Identity{member}},
		},
		CommitTime: fixedTime(),
	})
	if err != nil {
		t.Fatalf("ledger.New: %v", err)
	}
	return ls, kPub, kPriv
}

func makeSpeak(t *testing.T, priv ed25519.PrivateKey, pub ed25519.PublicKey, channel, content string, seq uint64) *eventsv1.EventEnvelope {
	t.Helper()
	payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: channel, Content: content})
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              1,
		SequencePerLedger: seq,
		EventType:         "Speak",
		Payload:           payload,
	}
	sig, _ := crypto.SignEnvelope(priv, env)
	env.Signature = sig
	return env
}

func runScenario(t *testing.T, ls *ledger.LedgerSet, pub ed25519.PublicKey, priv ed25519.PrivateKey) {
	t.Helper()
	for i := uint64(1); i <= 5; i++ {
		env := makeSpeak(t, priv, pub, "public", "msg", i)
		if _, err := ls.Commit(env, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
			t.Fatalf("commit %d: %v", i, err)
		}
	}
	for i := uint64(1); i <= 3; i++ {
		env := makeSpeak(t, priv, pub, "ch_07", "secret", i)
		if _, err := ls.Commit(env, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: "ch_07"}); err != nil {
			t.Fatalf("commit prv %d: %v", i, err)
		}
	}
}

func TestWriteAndReadRoundTrip(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newSet(t, pub)
	runScenario(t, ls, pub, priv)

	init := InitialState{ResearcherPubkeyHex: "deadbeef"}
	var buf bytes.Buffer
	hdr := Header{
		KernelVersion: "test",
		BuildHash:     "abc",
		ModelVersions: ModelVersions{
			AgentModel:            "meta-llama/Llama-3.1-8B-Instruct",
			AgentModelRevisionSHA: "0e9e39f249a16976918f6564b8830bc894c89659",
			AgentModelDtype:       "bf16",
			VLLMBuildSHA:          "v0.6.4",
			JudgeModel:            "Qwen/Qwen2.5-14B-Instruct",
			JudgeModelRevisionSHA: "deadbeef0123456789abcdef0123456789abcdef",
		},
	}
	if err := Write(&buf, hdr, ls, init); err != nil {
		t.Fatalf("Write: %v", err)
	}
	man, err := Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if man.SchemaVersion != SchemaVersion {
		t.Fatalf("schema version = %d, want %d", man.SchemaVersion, SchemaVersion)
	}
	if man.ModelVersions.AgentModel != "meta-llama/Llama-3.1-8B-Instruct" ||
		man.ModelVersions.AgentModelRevisionSHA != "0e9e39f249a16976918f6564b8830bc894c89659" ||
		man.ModelVersions.JudgeModel != "Qwen/Qwen2.5-14B-Instruct" {
		t.Fatalf("model_versions did not round-trip: %+v", man.ModelVersions)
	}
	if man.KernelVersion != "test" || man.BuildHash != "abc" {
		t.Fatalf("header drift: %+v", man)
	}
	if man.FinalChainHash != hex.EncodeToString(ls.ChainHead()) {
		t.Fatalf("final chain hash drift")
	}
	if len(man.Events) != len(ls.CommittedLog()) {
		t.Fatalf("event count = %d, want %d", len(man.Events), len(ls.CommittedLog()))
	}
}

func TestReplayMatchesOriginal(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	original, _, kPriv := newSet(t, pub)
	runScenario(t, original, pub, priv)

	var buf bytes.Buffer
	if err := Write(&buf, Header{KernelVersion: "test"}, original, InitialState{}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	man, err := Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	// fresh kernel with the same researcher and kernel key
	fresh, err := ledger.New(ledger.Config{
		Researcher: original.Researcher(),
		KernelPriv: kPriv,
		PrivateChannels: []ledger.PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []ledger.Identity{}},
		},
		CommitTime: fixedTime(),
	})
	if err != nil {
		t.Fatalf("fresh: %v", err)
	}
	got, err := Replay(man, fresh)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if !bytes.Equal(got, original.ChainHead()) {
		t.Fatalf("replay chain head drift")
	}
}

func TestReplayCatchesTamper(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	original, _, kPriv := newSet(t, pub)
	runScenario(t, original, pub, priv)

	var buf bytes.Buffer
	if err := Write(&buf, Header{KernelVersion: "test"}, original, InitialState{}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	man, err := Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	// Mutate the manifest's claimed chain head before replay.
	man.FinalChainHash = "00" + man.FinalChainHash[2:]
	fresh, err := ledger.New(ledger.Config{
		Researcher: original.Researcher(),
		KernelPriv: kPriv,
		PrivateChannels: []ledger.PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []ledger.Identity{}},
		},
		CommitTime: fixedTime(),
	})
	if err != nil {
		t.Fatalf("fresh: %v", err)
	}
	if _, err := Replay(man, fresh); err == nil {
		t.Fatal("Replay should fail when manifest's final chain hash is tampered")
	}
}

func TestReplayDeterministicAcrossManyRuns(t *testing.T) {
	// Experiment 003 motivator: replay the same manifest 50 times and
	// confirm every run produces the same chain head.
	pub, priv, _ := crypto.GenerateKey()
	original, _, kPriv := newSet(t, pub)
	runScenario(t, original, pub, priv)

	var buf bytes.Buffer
	if err := Write(&buf, Header{KernelVersion: "test"}, original, InitialState{}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	manBytes := buf.Bytes()
	var ref []byte
	for i := 0; i < 50; i++ {
		man, err := Read(bytes.NewReader(manBytes))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		fresh, err := ledger.New(ledger.Config{
			Researcher: original.Researcher(),
			KernelPriv: kPriv,
			PrivateChannels: []ledger.PrivateChannelConfig{
				{ChannelID: "ch_07", Members: []ledger.Identity{}},
			},
			CommitTime: fixedTime(),
		})
		if err != nil {
			t.Fatalf("fresh: %v", err)
		}
		got, err := Replay(man, fresh)
		if err != nil {
			t.Fatalf("Replay run %d: %v", i, err)
		}
		if ref == nil {
			ref = got
		} else if !bytes.Equal(ref, got) {
			t.Fatalf("run %d chain head drift: %x vs %x", i, ref, got)
		}
	}
}
