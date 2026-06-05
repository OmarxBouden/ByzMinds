package crypto

import (
	"bytes"
	"crypto/ed25519"
	"testing"

	"google.golang.org/protobuf/proto"

	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

func makeEnvelope(t *testing.T, priv ed25519.PrivateKey, pub ed25519.PublicKey) *eventsv1.EventEnvelope {
	t.Helper()
	speak := &eventsv1.Speak{ChannelId: "public", Content: "hello"}
	payload, err := CanonicalBytes(speak)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              7,
		SequencePerLedger: 3,
		EventType:         "Speak",
		Payload:           payload,
	}
	sig, err := SignEnvelope(priv, env)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	env.Signature = sig
	return env
}

func TestSignAndVerifyRoundTrip(t *testing.T) {
	pub, priv, err := GenerateKey()
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	env := makeEnvelope(t, priv, pub)
	if err := VerifyEnvelope(env); err != nil {
		t.Fatalf("verify failed on freshly signed envelope: %v", err)
	}
}

func TestVerifyRejectsTamperedPayload(t *testing.T) {
	pub, priv, _ := GenerateKey()
	env := makeEnvelope(t, priv, pub)
	// Mutate the payload after signing.
	env.Payload = []byte("totally different bytes")
	if err := VerifyEnvelope(env); err == nil {
		t.Fatal("verify must fail when payload is tampered")
	}
}

func TestVerifyRejectsTamperedTick(t *testing.T) {
	pub, priv, _ := GenerateKey()
	env := makeEnvelope(t, priv, pub)
	env.Tick++
	if err := VerifyEnvelope(env); err == nil {
		t.Fatal("verify must fail when tick is changed")
	}
}

func TestVerifyRejectsForeignKey(t *testing.T) {
	pub, priv, _ := GenerateKey()
	env := makeEnvelope(t, priv, pub)
	otherPub, _, _ := GenerateKey()
	env.EmitterPubkey = otherPub
	if err := VerifyEnvelope(env); err == nil {
		t.Fatal("verify must fail when pubkey is swapped")
	}
}

func TestCanonicalBytesAreStable(t *testing.T) {
	// Marshal the same message twice; bytes must be byte-identical.
	msg := &eventsv1.Speak{ChannelId: "ch_07", Content: "deterministic?"}
	a, err := CanonicalBytes(msg)
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	b, err := CanonicalBytes(msg)
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("canonical bytes drifted across marshal calls: %x vs %x", a, b)
	}
}

func TestSigningInputReconstruction(t *testing.T) {
	// SigningInputForEnvelope must produce the same bytes whether built
	// before signing or reconstructed from the envelope at verify time.
	pub, priv, _ := GenerateKey()
	speakPayload, _ := CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: "x"})
	si := &eventsv1.SigningInput{
		EmitterPubkey:     pub,
		Tick:              42,
		SequencePerLedger: 1,
		EventType:         "Speak",
		Payload:           speakPayload,
	}
	siBytes, _ := CanonicalBytes(si)

	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              42,
		SequencePerLedger: 1,
		EventType:         "Speak",
		Payload:           speakPayload,
	}
	envBytes, _ := CanonicalBytes(SigningInputForEnvelope(env))

	if !bytes.Equal(siBytes, envBytes) {
		t.Fatalf("reconstructed SigningInput drifted from original")
	}

	// And the signature matches under both paths.
	sigDirect := ed25519.Sign(priv, siBytes)
	if err := VerifyBytes(pub, envBytes, sigDirect); err != nil {
		t.Fatalf("sig verified against direct bytes must verify against reconstructed: %v", err)
	}
}

func TestVerifyBytesRejectsBadSig(t *testing.T) {
	pub, _, _ := GenerateKey()
	if err := VerifyBytes(pub, []byte("data"), make([]byte, SignatureSize)); err == nil {
		t.Fatal("VerifyBytes must reject zero signature")
	}
}

// Sanity check: the proto round-trip is itself stable. Belt-and-braces
// against an unintentional non-determinism regression in the runtime.
func TestEnvelopeRoundTripStable(t *testing.T) {
	pub, priv, _ := GenerateKey()
	env := makeEnvelope(t, priv, pub)
	a, _ := CanonicalBytes(env)
	b, _ := CanonicalBytes(env)
	if !bytes.Equal(a, b) {
		t.Fatal("envelope canonical bytes drifted")
	}
	// And it round-trips through unmarshal.
	clone := &eventsv1.EventEnvelope{}
	if err := proto.Unmarshal(a, clone); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if err := VerifyEnvelope(clone); err != nil {
		t.Fatalf("verify after roundtrip: %v", err)
	}
}
