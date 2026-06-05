// Package schema validates incoming envelopes and resolves them to a
// destination ledger. Two layers:
//
//  1. Envelope validation — non-empty pubkey/sig/event_type, bounded
//     payload, valid emitter pubkey length. Pure shape checks.
//  2. Payload validation — the bytes payload unmarshals into the typed
//     message named by event_type, and required fields are well-formed.
//
// Admissibility checks that depend on kernel state (phase, channel
// membership, current capabilities) live with the dispatcher in Step 2.
// This package is pure and stateless — given an envelope, it tells the
// kernel "your wire format is OK, route to ledger X" or "reject with
// reason Y". The kernel then layers state-dependent admissibility on top.
package schema

import (
	"errors"
	"fmt"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// MaxPayloadBytes caps payload size to keep one bad client from filling
// memory before admissibility runs. Tunable; 64 KiB is generous for any
// Stage A event.
const MaxPayloadBytes = 64 * 1024

// MaxSpeakContentBytes is the per-message content cap. Mirrors the
// "content length-bounded" admissibility note in the design.
const MaxSpeakContentBytes = 4 * 1024

// Event type names. Kept as constants so a typo in one place is a build
// error rather than a runtime mystery.
const (
	TypeSpeak             = "Speak"
	TypeVote              = "Vote"
	TypeOpenChannelReq    = "OpenChannelReq"
	TypeCloseChannelReq   = "CloseChannelReq"
	TypeRequestCapability = "RequestCapability"
	TypeDropCapability    = "DropCapability"
	TypeYield             = "Yield"
	TypeDeclareIntent     = "DeclareIntent"
)

// ErrEnvelope is returned for any envelope-shape failure.
var ErrEnvelope = errors.New("schema: envelope invalid")

// ErrPayload is returned when the payload bytes do not parse as the
// declared event type, or required payload fields are missing.
var ErrPayload = errors.New("schema: payload invalid")

// ErrUnknownEventType is returned when event_type is not a Stage A type.
var ErrUnknownEventType = errors.New("schema: unknown event_type")

// Validated is the result of a successful Validate call: the parsed
// payload (concrete proto.Message) and the resolved destination ledger.
type Validated struct {
	Payload     proto.Message
	Destination ledger.Destination
}

// PublicChannelID is the conventional channel id that routes Speak/Vote
// to L_pub. Any other id routes to L_prv[id].
const PublicChannelID = "public"

// Validate runs envelope-shape and payload checks and resolves the
// destination ledger. It does not do signature verification (that lives
// in the crypto package) or stateful admissibility (Step 2).
func Validate(env *eventsv1.EventEnvelope) (*Validated, error) {
	if env == nil {
		return nil, fmt.Errorf("%w: nil envelope", ErrEnvelope)
	}
	if len(env.GetEmitterPubkey()) != crypto.PublicKeySize {
		return nil, fmt.Errorf("%w: emitter_pubkey must be %d bytes", ErrEnvelope, crypto.PublicKeySize)
	}
	if len(env.GetSignature()) != crypto.SignatureSize {
		return nil, fmt.Errorf("%w: signature must be %d bytes", ErrEnvelope, crypto.SignatureSize)
	}
	if env.GetEventType() == "" {
		return nil, fmt.Errorf("%w: event_type is required", ErrEnvelope)
	}
	if env.GetSequencePerLedger() == 0 {
		return nil, fmt.Errorf("%w: sequence_per_ledger must be >= 1", ErrEnvelope)
	}
	if len(env.GetPayload()) > MaxPayloadBytes {
		return nil, fmt.Errorf("%w: payload too large (%d > %d)", ErrEnvelope, len(env.GetPayload()), MaxPayloadBytes)
	}

	switch env.GetEventType() {
	case TypeSpeak:
		return validateSpeak(env)
	case TypeVote:
		return validateVote(env)
	case TypeOpenChannelReq:
		return validateOpenChannel(env)
	case TypeCloseChannelReq:
		return validateCloseChannel(env)
	case TypeRequestCapability:
		return validateRequestCap(env)
	case TypeDropCapability:
		return validateDropCap(env)
	case TypeYield:
		return validateYield(env)
	case TypeDeclareIntent:
		return validateDeclareIntent(env)
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownEventType, env.GetEventType())
	}
}

func unmarshalPayload(env *eventsv1.EventEnvelope, into proto.Message) error {
	if err := proto.Unmarshal(env.GetPayload(), into); err != nil {
		return fmt.Errorf("%w: %v", ErrPayload, err)
	}
	return nil
}

func validateSpeak(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.Speak{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetChannelId() == "" {
		return nil, fmt.Errorf("%w: Speak.channel_id required", ErrPayload)
	}
	if len(msg.GetContent()) > MaxSpeakContentBytes {
		return nil, fmt.Errorf("%w: Speak.content too long", ErrPayload)
	}
	dest := destinationForChannel(msg.GetChannelId())
	return &Validated{Payload: msg, Destination: dest}, nil
}

func validateVote(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.Vote{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetOption() == "" {
		return nil, fmt.Errorf("%w: Vote.option required", ErrPayload)
	}
	// Votes always commit publicly; scenario admissibility (phase, enum)
	// is layered on top in Step 2.
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}}, nil
}

func validateOpenChannel(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.OpenChannelReq{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if len(msg.GetProposedMembers()) < 2 {
		return nil, fmt.Errorf("%w: OpenChannelReq.proposed_members must have >= 2", ErrPayload)
	}
	// Channel-open requests are control events: route to L_ctrl. The
	// handler will mint the actual L_prv ledger in Step 2.
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}}, nil
}

func validateCloseChannel(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.CloseChannelReq{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetChannelId() == "" {
		return nil, fmt.Errorf("%w: CloseChannelReq.channel_id required", ErrPayload)
	}
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}}, nil
}

func validateRequestCap(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.RequestCapability{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetCapId() == "" {
		return nil, fmt.Errorf("%w: RequestCapability.cap_id required", ErrPayload)
	}
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}}, nil
}

func validateDropCap(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.DropCapability{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetCapId() == "" {
		return nil, fmt.Errorf("%w: DropCapability.cap_id required", ErrPayload)
	}
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_CTRL}}, nil
}

func validateYield(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.Yield{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	// Yield is always admissible and writes to L_pub so other agents see it.
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}}, nil
}

func validateDeclareIntent(env *eventsv1.EventEnvelope) (*Validated, error) {
	msg := &eventsv1.DeclareIntent{}
	if err := unmarshalPayload(env, msg); err != nil {
		return nil, err
	}
	if msg.GetContent() == "" {
		return nil, fmt.Errorf("%w: DeclareIntent.content required", ErrPayload)
	}
	return &Validated{Payload: msg, Destination: ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_COG_ELI}}, nil
}

// destinationForChannel maps a channel id string to a ledger destination.
// The conventional id "public" lands on L_pub; anything else routes to
// L_prv[id] and is rejected at commit time if the channel does not exist.
func destinationForChannel(channelID string) ledger.Destination {
	if channelID == PublicChannelID {
		return ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB}
	}
	return ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PRV, ChannelID: channelID}
}
