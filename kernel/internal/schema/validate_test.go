package schema

import (
	"crypto/ed25519"
	"errors"
	"strings"
	"testing"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

func mustKey(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	return pub, priv
}

func envFor(t *testing.T, eventType string, payload []byte) *eventsv1.EventEnvelope {
	t.Helper()
	pub, priv := mustKey(t)
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     pub,
		Tick:              1,
		SequencePerLedger: 1,
		EventType:         eventType,
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(priv, env)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	env.Signature = sig
	return env
}

func TestValidateSpeakOnPublicRoutesToLPub(t *testing.T) {
	payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: "hi"})
	env := envFor(t, TypeSpeak, payload)
	v, err := Validate(env)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if v.Destination.LedgerID != ledgerv1.LedgerID_LEDGER_ID_L_PUB {
		t.Fatalf("dest = %v, want L_PUB", v.Destination.LedgerID)
	}
}

func TestValidateSpeakOnPrivateRoutesToLPrv(t *testing.T) {
	payload, _ := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "ch_07", Content: "hi"})
	env := envFor(t, TypeSpeak, payload)
	v, err := Validate(env)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if v.Destination.LedgerID != ledgerv1.LedgerID_LEDGER_ID_L_PRV || v.Destination.ChannelID != "ch_07" {
		t.Fatalf("dest = %+v, want L_PRV[ch_07]", v.Destination)
	}
}

func TestValidateDeclareIntentRoutesToLCogEli(t *testing.T) {
	payload, _ := crypto.CanonicalBytes(&eventsv1.DeclareIntent{Content: "I voted yes because the data supports it."})
	env := envFor(t, TypeDeclareIntent, payload)
	v, err := Validate(env)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if v.Destination.LedgerID != ledgerv1.LedgerID_LEDGER_ID_L_COG_ELI {
		t.Fatalf("dest = %v, want L_COG_ELI", v.Destination.LedgerID)
	}
}

func TestValidateControlEventsRouteToLCtrl(t *testing.T) {
	cases := []struct {
		name    string
		evtType string
		payload []byte
	}{
		{
			name:    "open channel",
			evtType: TypeOpenChannelReq,
			payload: must(crypto.CanonicalBytes(&eventsv1.OpenChannelReq{ProposedMembers: []string{"a", "b"}})),
		},
		{
			name:    "close channel",
			evtType: TypeCloseChannelReq,
			payload: must(crypto.CanonicalBytes(&eventsv1.CloseChannelReq{ChannelId: "ch_07"})),
		},
		{
			name:    "request capability",
			evtType: TypeRequestCapability,
			payload: must(crypto.CanonicalBytes(&eventsv1.RequestCapability{CapId: "search", Justification: "needed"})),
		},
		{
			name:    "drop capability",
			evtType: TypeDropCapability,
			payload: must(crypto.CanonicalBytes(&eventsv1.DropCapability{CapId: "search"})),
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			env := envFor(t, c.evtType, c.payload)
			v, err := Validate(env)
			if err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if v.Destination.LedgerID != ledgerv1.LedgerID_LEDGER_ID_L_CTRL {
				t.Fatalf("dest = %v, want L_CTRL", v.Destination.LedgerID)
			}
		})
	}
}

func TestValidateRejectsUnknownEventType(t *testing.T) {
	env := envFor(t, "Wat", nil)
	_, err := Validate(env)
	if !errors.Is(err, ErrUnknownEventType) {
		t.Fatalf("err = %v, want ErrUnknownEventType", err)
	}
}

func TestValidateRejectsBadEnvelopeShape(t *testing.T) {
	pub, priv := mustKey(t)
	cases := []struct {
		name string
		mut  func(*eventsv1.EventEnvelope)
	}{
		{"empty event_type", func(e *eventsv1.EventEnvelope) { e.EventType = "" }},
		{"zero seq", func(e *eventsv1.EventEnvelope) { e.SequencePerLedger = 0 }},
		{"short pubkey", func(e *eventsv1.EventEnvelope) { e.EmitterPubkey = []byte{1, 2, 3} }},
		{"short signature", func(e *eventsv1.EventEnvelope) { e.Signature = []byte{0, 1, 2} }},
		{"oversize payload", func(e *eventsv1.EventEnvelope) { e.Payload = make([]byte, MaxPayloadBytes+1) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			payload, _ := crypto.CanonicalBytes(&eventsv1.Yield{Reason: "x"})
			env := &eventsv1.EventEnvelope{
				EmitterPubkey:     pub,
				Tick:              1,
				SequencePerLedger: 1,
				EventType:         TypeYield,
				Payload:           payload,
			}
			sig, _ := crypto.SignEnvelope(priv, env)
			env.Signature = sig
			c.mut(env)
			_, err := Validate(env)
			if !errors.Is(err, ErrEnvelope) {
				t.Fatalf("err = %v, want ErrEnvelope", err)
			}
		})
	}
}

func TestValidateRejectsBadPayloads(t *testing.T) {
	cases := []struct {
		name    string
		evtType string
		payload []byte
		errIs   error
	}{
		{
			name:    "speak missing channel",
			evtType: TypeSpeak,
			payload: must(crypto.CanonicalBytes(&eventsv1.Speak{Content: "x"})),
			errIs:   ErrPayload,
		},
		{
			name:    "speak too long",
			evtType: TypeSpeak,
			payload: must(crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: strings.Repeat("a", MaxSpeakContentBytes+1)})),
			errIs:   ErrPayload,
		},
		{
			name:    "vote missing option",
			evtType: TypeVote,
			payload: must(crypto.CanonicalBytes(&eventsv1.Vote{})),
			errIs:   ErrPayload,
		},
		{
			name:    "open channel too few members",
			evtType: TypeOpenChannelReq,
			payload: must(crypto.CanonicalBytes(&eventsv1.OpenChannelReq{ProposedMembers: []string{"only-me"}})),
			errIs:   ErrPayload,
		},
		{
			name:    "garbage payload bytes",
			evtType: TypeSpeak,
			payload: []byte{0xff, 0xfe, 0xfd},
			errIs:   ErrPayload,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			env := envFor(t, c.evtType, c.payload)
			_, err := Validate(env)
			if !errors.Is(err, c.errIs) {
				t.Fatalf("err = %v, want %v", err, c.errIs)
			}
		})
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
