package api

import (
	"context"
	"crypto/ed25519"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	kernelv1 "github.com/byzminds/byzminds/proto/kernelv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

const bufSize = 1024 * 1024

type rig struct {
	t       *testing.T
	listen  *bufconn.Listener
	grpcSrv *grpc.Server
	ls      *ledger.LedgerSet
	srv     *Server

	researcherPub  ed25519.PublicKey
	researcherPriv ed25519.PrivateKey
	memberPub      ed25519.PublicKey
	memberPriv     ed25519.PrivateKey
}

func setupRig(t *testing.T) *rig {
	t.Helper()
	rPub, rPriv, _ := crypto.GenerateKey()
	mPub, mPriv, _ := crypto.GenerateKey()
	_, kPriv, _ := crypto.GenerateKey()
	ls, err := ledger.New(ledger.Config{
		Researcher: rPub,
		KernelPriv: kPriv,
		PrivateChannels: []ledger.PrivateChannelConfig{
			{ChannelID: "ch_07", Members: []ledger.Identity{mPub}},
		},
		CommitTime: func() uint64 { return 1_700_000_000_000_000_000 },
	})
	if err != nil {
		t.Fatalf("ledger.New: %v", err)
	}
	srv := New(ls)

	gs := grpc.NewServer()
	srv.Register(gs)
	lis := bufconn.Listen(bufSize)
	go func() {
		if err := gs.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			t.Errorf("grpc serve: %v", err)
		}
	}()
	t.Cleanup(func() {
		gs.Stop()
	})
	return &rig{
		t: t, listen: lis, grpcSrv: gs, ls: ls, srv: srv,
		researcherPub: rPub, researcherPriv: rPriv,
		memberPub: mPub, memberPriv: mPriv,
	}
}

func (r *rig) client(t *testing.T) kernelv1.KernelClient {
	t.Helper()
	dialer := func(_ context.Context, _ string) (net.Conn, error) { return r.listen.Dial() }
	conn, err := grpc.NewClient("passthrough://bufconn", grpc.WithContextDialer(dialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return kernelv1.NewKernelClient(conn)
}

func signedSpeak(t *testing.T, priv ed25519.PrivateKey, pub ed25519.PublicKey, channel, content string, tick, seq uint64) *eventsv1.EventEnvelope {
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

func TestSubmitEventCommitsValidEnvelope(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	env := signedSpeak(t, r.memberPriv, r.memberPub, "public", "hello", 1, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	rec, err := cli.SubmitEvent(ctx, env)
	if err != nil {
		t.Fatalf("SubmitEvent: %v", err)
	}
	if !rec.GetCommitted() {
		t.Fatalf("not committed: %s", rec.GetRejectionReason())
	}
	if rec.GetSequencePerLedger() != 1 || rec.GetGlobalCommitSeq() != 1 {
		t.Fatalf("seqs = (%d, %d), want (1, 1)", rec.GetSequencePerLedger(), rec.GetGlobalCommitSeq())
	}
	if len(rec.GetChainHash()) != 32 {
		t.Fatalf("chain_hash len = %d, want 32", len(rec.GetChainHash()))
	}
}

func TestSubmitEventRejectsBadSignature(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	env := signedSpeak(t, r.memberPriv, r.memberPub, "public", "hi", 1, 1)
	env.Signature[0] ^= 0x01 // flip a bit
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	rec, err := cli.SubmitEvent(ctx, env)
	if err != nil {
		t.Fatalf("SubmitEvent: %v", err)
	}
	if rec.GetCommitted() {
		t.Fatal("committed despite bad signature")
	}
	if rec.GetRejectionReason() == "" {
		t.Fatal("rejection_reason empty")
	}
}

func TestSubmitEventRejectsSequenceMismatch(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	// First commit takes seq 1; second tries to claim 5 instead of 2.
	first := signedSpeak(t, r.memberPriv, r.memberPub, "public", "a", 1, 1)
	bad := signedSpeak(t, r.memberPriv, r.memberPub, "public", "b", 1, 5)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := cli.SubmitEvent(ctx, first); err != nil {
		t.Fatalf("first SubmitEvent: %v", err)
	}
	rec, err := cli.SubmitEvent(ctx, bad)
	if err != nil {
		t.Fatalf("SubmitEvent: %v", err)
	}
	if rec.GetCommitted() {
		t.Fatal("kernel accepted out-of-order seq")
	}
}

func TestGetViewStreamsAccessibleEvents(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	// One public event, one private to ch_07.
	if _, err := cli.SubmitEvent(ctx, signedSpeak(t, r.memberPriv, r.memberPub, "public", "p1", 1, 1)); err != nil {
		t.Fatalf("submit pub: %v", err)
	}
	if _, err := cli.SubmitEvent(ctx, signedSpeak(t, r.memberPriv, r.memberPub, "ch_07", "s1", 1, 1)); err != nil {
		t.Fatalf("submit prv: %v", err)
	}

	// Member of ch_07 should see both.
	view := mustView(t, cli, r.memberPub, r.memberPriv, 0)
	if got := len(view); got != 2 {
		t.Fatalf("member view len = %d, want 2", got)
	}

	// Outsider sees only public.
	outPub, outPriv, _ := crypto.GenerateKey()
	view = mustView(t, cli, outPub, outPriv, 0)
	if got := len(view); got != 1 {
		t.Fatalf("outsider view len = %d, want 1", got)
	}
	if view[0].GetEvent().GetLedgerId() != ledgerv1.LedgerID_LEDGER_ID_L_PUB {
		t.Fatalf("outsider saw non-public event")
	}
}

func TestGetViewRejectsBadSignature(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	pub, priv, _ := crypto.GenerateKey()
	bad := &kernelv1.ViewRequest{
		ReaderPubkey: pub,
		FromTick:     0,
		Signature:    ed25519.Sign(priv, []byte("not the right canonical bytes")),
	}
	stream, err := cli.GetView(ctx, bad)
	if err != nil {
		t.Fatalf("GetView call: %v", err)
	}
	_, err = stream.Recv()
	if err == nil {
		t.Fatal("GetView accepted bad signature")
	}
}

func TestGetViewRespectsFromTick(t *testing.T) {
	r := setupRig(t)
	cli := r.client(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := cli.SubmitEvent(ctx, signedSpeak(t, r.memberPriv, r.memberPub, "public", "early", 1, 1)); err != nil {
		t.Fatalf("submit early: %v", err)
	}
	if _, err := cli.SubmitEvent(ctx, signedSpeak(t, r.memberPriv, r.memberPub, "public", "late", 5, 2)); err != nil {
		t.Fatalf("submit late: %v", err)
	}
	view := mustView(t, cli, r.memberPub, r.memberPriv, 5)
	if got := len(view); got != 1 {
		t.Fatalf("from_tick=5 view len = %d, want 1", got)
	}
	if view[0].GetEvent().GetEnvelope().GetTick() != 5 {
		t.Fatalf("got tick %d, want 5", view[0].GetEvent().GetEnvelope().GetTick())
	}
}

func mustView(t *testing.T, cli kernelv1.KernelClient, pub ed25519.PublicKey, priv ed25519.PrivateKey, fromTick uint64) []*kernelv1.EventView {
	t.Helper()
	signing := ViewRequestSigningBytes(pub, fromTick)
	sig := ed25519.Sign(priv, signing)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	stream, err := cli.GetView(ctx, &kernelv1.ViewRequest{ReaderPubkey: pub, FromTick: fromTick, Signature: sig})
	if err != nil {
		t.Fatalf("GetView: %v", err)
	}
	var out []*kernelv1.EventView
	for {
		v, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("stream.Recv: %v", err)
		}
		out = append(out, v)
	}
	return out
}
