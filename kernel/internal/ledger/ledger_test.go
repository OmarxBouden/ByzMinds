package ledger

import (
	"bytes"
	"crypto/ed25519"
	"errors"
	"testing"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// fixedTime returns a deterministic CommitTime for replay-stable tests.
func fixedTime() CommitTime { return func() uint64 { return 1_700_000_000_000_000_000 } }

// newTestSet returns a LedgerSet with one private channel ch_07
// containing the supplied member identity.
func newTestSet(t *testing.T, member ed25519.PublicKey) (*LedgerSet, ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	researcherPub, _, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	kPub, kPriv, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	cfg := Config{
		Researcher: researcherPub,
		KernelPriv: kPriv,
		PrivateChannels: []PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []Identity{member}},
		},
		CommitTime: fixedTime(),
	}
	ls, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ls, kPub, kPriv
}

// signed envelope helper.
func makeSpeak(t *testing.T, priv ed25519.PrivateKey, pub ed25519.PublicKey, channel, content string, tick, seq uint64) *eventsv1.EventEnvelope {
	t.Helper()
	payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: channel, Content: content})
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              tick,
		SequencePerLedger: seq,
		EventType:         "Speak",
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(priv, env)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	env.Signature = sig
	return env
}

func TestTwoEmittersCanShareSeqOnSameLedger(t *testing.T) {
	// Per-(emitter, ledger) sequencing: emitter A's seq=1 and emitter B's
	// seq=1 on L_pub must both commit. Their global_commit_seqs differ.
	pubA, privA, _ := crypto.GenerateKey()
	pubB, privB, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pubA)

	envA := makeSpeak(t, privA, pubA, "public", "from-A", 1, 1)
	envB := makeSpeak(t, privB, pubB, "public", "from-B", 1, 1)

	cA, err := ls.Commit(envA, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if err != nil {
		t.Fatalf("commit A: %v", err)
	}
	cB, err := ls.Commit(envB, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if err != nil {
		t.Fatalf("commit B (must succeed under per-emitter seqs): %v", err)
	}
	if cA.GetEnvelope().GetSequencePerLedger() != 1 || cB.GetEnvelope().GetSequencePerLedger() != 1 {
		t.Fatalf("both should claim seq 1; got A=%d B=%d",
			cA.GetEnvelope().GetSequencePerLedger(), cB.GetEnvelope().GetSequencePerLedger())
	}
	if cA.GetGlobalCommitSeq() == cB.GetGlobalCommitSeq() {
		t.Fatalf("global_commit_seq must differ; both = %d", cA.GetGlobalCommitSeq())
	}
}

func TestEmitterSequenceIsPerDestination(t *testing.T) {
	// Same emitter across two destinations: each gets its own counter
	// starting at 1.
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)

	envPub := makeSpeak(t, priv, pub, "public", "p", 1, 1)
	if _, err := ls.Commit(envPub, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
		t.Fatalf("commit pub: %v", err)
	}
	// Now seq=1 on L_prv:ch_07 must succeed — different destination, fresh counter.
	envPrv := makeSpeak(t, priv, pub, "ch_07", "s", 1, 1)
	if _, err := ls.Commit(envPrv, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: "ch_07"}); err != nil {
		t.Fatalf("commit prv (must succeed): %v", err)
	}
}

func TestSameEmitterCannotRepeatSeq(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	first := makeSpeak(t, priv, pub, "public", "1", 1, 1)
	if _, err := ls.Commit(first, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
		t.Fatalf("commit first: %v", err)
	}
	dup := makeSpeak(t, priv, pub, "public", "1-again", 1, 1)
	_, err := ls.Commit(dup, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if !errors.Is(err, ErrSequenceMismatch) {
		t.Fatalf("err = %v, want ErrSequenceMismatch (same emitter cannot repeat seq)", err)
	}
}

func TestCommitFirstEventGetsSeq1(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	env := makeSpeak(t, priv, pub, "public", "hi", 1, 1)
	got, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if err != nil {
		t.Fatalf("Commit: %v", err)
	}
	if got.GetEnvelope().GetSequencePerLedger() != 1 {
		t.Errorf("first event seq = %d, want 1", got.GetEnvelope().GetSequencePerLedger())
	}
	if got.GetGlobalCommitSeq() != 1 {
		t.Errorf("global_commit_seq = %d, want 1", got.GetGlobalCommitSeq())
	}
	if !bytes.Equal(got.GetPrevChainHash(), make([]byte, 32)) {
		t.Errorf("first prev_chain_hash should be 32 zero bytes, got %x", got.GetPrevChainHash())
	}
	if len(got.GetChainHash()) != 32 {
		t.Errorf("chain_hash should be 32 bytes, got %d", len(got.GetChainHash()))
	}
}

func TestCommitRejectsSeqMismatch(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	env := makeSpeak(t, priv, pub, "public", "hi", 1, 5) // wrong seq, should be 1
	_, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if !errors.Is(err, ErrSequenceMismatch) {
		t.Fatalf("Commit err = %v, want ErrSequenceMismatch", err)
	}
}

func TestCommitChainExtendsLinearly(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	prevHead := ls.ChainHead()
	for i := uint64(1); i <= 5; i++ {
		env := makeSpeak(t, priv, pub, "public", "msg", 1, i)
		c, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
		if err != nil {
			t.Fatalf("Commit %d: %v", i, err)
		}
		if !bytes.Equal(c.GetPrevChainHash(), prevHead) {
			t.Fatalf("event %d prev_chain_hash does not match prior head", i)
		}
		prevHead = c.GetChainHash()
	}
	// Final head matches LedgerSet's chain head.
	if !bytes.Equal(ls.ChainHead(), prevHead) {
		t.Fatal("final ChainHead() drifted from per-event chain_hash")
	}
}

func TestCommitInterleavedAcrossLedgers(t *testing.T) {
	memberPub, memberPriv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, memberPub)
	// Three commits: L_pub seq 1, L_prv:ch_07 seq 1, L_pub seq 2.
	e1 := makeSpeak(t, memberPriv, memberPub, "public", "a", 1, 1)
	e2 := makeSpeak(t, memberPriv, memberPub, "ch_07", "b", 1, 1)
	e3 := makeSpeak(t, memberPriv, memberPub, "public", "c", 1, 2)
	c1, err := ls.Commit(e1, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if err != nil {
		t.Fatalf("commit 1: %v", err)
	}
	c2, err := ls.Commit(e2, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: "ch_07"})
	if err != nil {
		t.Fatalf("commit 2: %v", err)
	}
	c3, err := ls.Commit(e3, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
	if err != nil {
		t.Fatalf("commit 3: %v", err)
	}
	if c1.GetGlobalCommitSeq() != 1 || c2.GetGlobalCommitSeq() != 2 || c3.GetGlobalCommitSeq() != 3 {
		t.Fatalf("global_commit_seq mismatch: got %d, %d, %d", c1.GetGlobalCommitSeq(), c2.GetGlobalCommitSeq(), c3.GetGlobalCommitSeq())
	}
	if !bytes.Equal(c2.GetPrevChainHash(), c1.GetChainHash()) ||
		!bytes.Equal(c3.GetPrevChainHash(), c2.GetChainHash()) {
		t.Fatal("chain not linear across ledgers")
	}
}

func TestCommitRejectsUnknownPrivateChannel(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	env := makeSpeak(t, priv, pub, "ch_999", "x", 1, 1)
	_, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: "ch_999"})
	if !errors.Is(err, ErrUnknownChannel) {
		t.Fatalf("err = %v, want ErrUnknownChannel", err)
	}
}

func TestCommitRejectsInvalidLedger(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	env := makeSpeak(t, priv, pub, "public", "x", 1, 1)
	_, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_UNSPECIFIED})
	if !errors.Is(err, ErrInvalidLedger) {
		t.Fatalf("err = %v, want ErrInvalidLedger", err)
	}
}

func TestAccessPolicyEnforcement(t *testing.T) {
	memberPub, _, _ := crypto.GenerateKey()
	outsiderPub, _, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, memberPub)

	// Public ledger: anyone reads.
	if !ls.Pub().CanRead(outsiderPub) {
		t.Error("L_pub should be readable by outsider")
	}
	// Cog/ctrl: researcher only.
	if ls.CogInd().CanRead(outsiderPub) {
		t.Error("L_cog_ind must not be readable by outsider")
	}
	if ls.CogEli().CanRead(memberPub) {
		t.Error("L_cog_eli must not be readable by channel member (only researcher)")
	}
	if ls.Ctrl().CanRead(memberPub) {
		t.Error("L_ctrl must not be readable by channel member")
	}
	// Private channel: members only.
	if !ls.Prv("ch_07").CanRead(memberPub) {
		t.Error("L_prv:ch_07 must be readable by member")
	}
	if ls.Prv("ch_07").CanRead(outsiderPub) {
		t.Error("L_prv:ch_07 must not be readable by outsider")
	}
}

func TestSnapshotAndIterate(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	ls, _, _ := newTestSet(t, pub)
	for i := uint64(1); i <= 4; i++ {
		env := makeSpeak(t, priv, pub, "public", "m", 1, i)
		if _, err := ls.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
			t.Fatalf("commit: %v", err)
		}
	}
	if got := ls.Pub().Len(); got != 4 {
		t.Fatalf("Len = %d, want 4", got)
	}
	snap := ls.Pub().Snapshot()
	if len(snap) != 4 {
		t.Fatalf("Snapshot len = %d, want 4", len(snap))
	}
	// Iterate from seq 3 → expect 2 events.
	var got []uint64
	for e := range ls.Pub().Iterate(3) {
		got = append(got, e.GetEnvelope().GetSequencePerLedger())
	}
	if len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Fatalf("Iterate(3) yielded %v, want [3 4]", got)
	}
}

func TestReplayHappyPath(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	original, _, kPriv := newTestSet(t, pub)
	for i := uint64(1); i <= 5; i++ {
		env := makeSpeak(t, priv, pub, "public", "m", 1, i)
		if _, err := original.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
			t.Fatalf("commit: %v", err)
		}
	}
	// Build a fresh kernel with the same researcher and kernel key.
	replay := mustNewLikeOriginal(t, original.researcher, kPriv)
	for _, c := range original.CommittedLog() {
		if err := replay.AppendReplay(c); err != nil {
			t.Fatalf("AppendReplay: %v", err)
		}
	}
	if !bytes.Equal(replay.ChainHead(), original.ChainHead()) {
		t.Fatalf("replay ChainHead differs:\noriginal=%x\nreplay  =%x", original.ChainHead(), replay.ChainHead())
	}
}

func TestReplayDetectsTamper(t *testing.T) {
	pub, priv, _ := crypto.GenerateKey()
	original, _, kPriv := newTestSet(t, pub)
	for i := uint64(1); i <= 3; i++ {
		env := makeSpeak(t, priv, pub, "public", "m", 1, i)
		if _, err := original.Commit(env, Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}); err != nil {
			t.Fatalf("commit: %v", err)
		}
	}
	log := original.CommittedLog()
	// Tamper with the second event's payload.
	log[1].Envelope.Payload = []byte("tampered")
	replay := mustNewLikeOriginal(t, original.researcher, kPriv)
	for i, c := range log {
		err := replay.AppendReplay(c)
		if i < 1 && err != nil {
			t.Fatalf("good event rejected: %v", err)
		}
		if i == 1 && err == nil {
			t.Fatal("replay accepted tampered event; chain hash check missed")
		}
		if err != nil {
			break
		}
	}
}

func mustNewLikeOriginal(t *testing.T, researcher ed25519.PublicKey, kPriv ed25519.PrivateKey) *LedgerSet {
	t.Helper()
	ls, err := New(Config{
		Researcher: researcher,
		KernelPriv: kPriv,
		PrivateChannels: []PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []Identity{}},
		},
		CommitTime: fixedTime(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ls
}
