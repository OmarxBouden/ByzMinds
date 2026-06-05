// Package crypto wraps Ed25519 sign/verify and defines the canonical bytes
// the kernel signs and verifies against.
//
// Two pieces of canonical content matter:
//
//   - SigningInput: what the *emitter* signs to produce
//     EventEnvelope.signature. Reconstructed from the envelope at verify time
//     so the signature is bound to (pubkey, tick, sequence, event_type,
//     payload), independent of how the envelope itself was framed on the wire.
//   - ChainInput: what the *kernel* hashes to produce CommittedEvent.chain_hash
//     and signs to produce CommittedEvent.kernel_signature. Lives in the
//     ledger package, but uses CanonicalBytes from here.
package crypto

import (
	"crypto/ed25519"
	"errors"
	"fmt"

	"google.golang.org/protobuf/proto"

	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

// PublicKeySize and PrivateKeySize match the standard Ed25519 sizes.
const (
	PublicKeySize  = ed25519.PublicKeySize
	PrivateKeySize = ed25519.PrivateKeySize
	SignatureSize  = ed25519.SignatureSize
)

// CanonicalBytes returns the deterministic protobuf marshaling of m. Used
// everywhere a hash or signature must be reproducible across runs.
//
// The Deterministic option fixes map-field ordering; SigningInput and
// ChainInput contain no maps, so the option is belt-and-braces.
func CanonicalBytes(m proto.Message) ([]byte, error) {
	return proto.MarshalOptions{Deterministic: true}.Marshal(m)
}

// SigningInputForEnvelope reconstructs the SigningInput that an emitter
// would have signed for the given envelope.
func SigningInputForEnvelope(env *eventsv1.EventEnvelope) *eventsv1.SigningInput {
	return &eventsv1.SigningInput{
		EmitterPubkey:     env.GetEmitterPubkey(),
		Tick:              env.GetTick(),
		SequencePerLedger: env.GetSequencePerLedger(),
		EventType:         env.GetEventType(),
		Payload:           env.GetPayload(),
	}
}

// SignEnvelope computes the canonical signing bytes for env and signs them
// with priv. The caller is expected to assign the returned signature into
// env.Signature.
func SignEnvelope(priv ed25519.PrivateKey, env *eventsv1.EventEnvelope) ([]byte, error) {
	if len(priv) != PrivateKeySize {
		return nil, fmt.Errorf("crypto: invalid private key length %d", len(priv))
	}
	bytes, err := CanonicalBytes(SigningInputForEnvelope(env))
	if err != nil {
		return nil, fmt.Errorf("crypto: marshal signing input: %w", err)
	}
	return ed25519.Sign(priv, bytes), nil
}

// VerifyEnvelope checks env.Signature against the canonical signing bytes
// reconstructed from env, using env.EmitterPubkey. Returns nil on success.
func VerifyEnvelope(env *eventsv1.EventEnvelope) error {
	if env == nil {
		return errors.New("crypto: nil envelope")
	}
	if len(env.GetEmitterPubkey()) != PublicKeySize {
		return fmt.Errorf("crypto: invalid emitter pubkey length %d", len(env.GetEmitterPubkey()))
	}
	if len(env.GetSignature()) != SignatureSize {
		return fmt.Errorf("crypto: invalid signature length %d", len(env.GetSignature()))
	}
	bytes, err := CanonicalBytes(SigningInputForEnvelope(env))
	if err != nil {
		return fmt.Errorf("crypto: marshal signing input: %w", err)
	}
	if !ed25519.Verify(env.GetEmitterPubkey(), bytes, env.GetSignature()) {
		return errors.New("crypto: signature verification failed")
	}
	return nil
}

// SignBytes signs raw bytes with priv. Used by the kernel to sign a
// chain_hash; callers handle whatever canonicalization is appropriate
// upstream.
func SignBytes(priv ed25519.PrivateKey, msg []byte) ([]byte, error) {
	if len(priv) != PrivateKeySize {
		return nil, fmt.Errorf("crypto: invalid private key length %d", len(priv))
	}
	return ed25519.Sign(priv, msg), nil
}

// VerifyBytes verifies a signature over raw bytes against pub.
func VerifyBytes(pub ed25519.PublicKey, msg, sig []byte) error {
	if len(pub) != PublicKeySize {
		return fmt.Errorf("crypto: invalid public key length %d", len(pub))
	}
	if len(sig) != SignatureSize {
		return fmt.Errorf("crypto: invalid signature length %d", len(sig))
	}
	if !ed25519.Verify(pub, msg, sig) {
		return errors.New("crypto: signature verification failed")
	}
	return nil
}

// GenerateKey is a thin wrapper around ed25519.GenerateKey for tests and
// the kernel binary. Callers may pass a deterministic io.Reader (e.g.
// math/rand seeded) when reproducibility matters.
func GenerateKey() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	return ed25519.GenerateKey(nil)
}
