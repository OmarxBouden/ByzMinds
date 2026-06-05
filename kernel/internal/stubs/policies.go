// Package stubs holds the three reference stub-agent policies (echo,
// mirror, silent) wrapped as scheduler.AgentTransport implementations.
// Per the Step 2 brief judgment call #3, stubs are permanent: they
// become the kernel's regression suite, exercised on every code change.
//
// Each policy:
//   - Reads the View it was just handed.
//   - Picks a tool from view.available_tools deterministically.
//   - Builds + signs the matching envelope with the agent's key,
//     using the kernel's NextSeqFor query to mint the right seq.
//   - Returns the signed envelope.
//
// Stubs deliberately have no network surface in this package — they
// embed a *ledger.LedgerSet so SequencePerLedger can be claimed without
// a kernel round-trip. The byzminds-stub-agent binary wraps the same
// policies behind a gRPC client for cross-process testing.
package stubs

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"strings"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

// Policy is the named stub behavior. The scenario YAML's agent.stub_policy
// field maps to one of these via Lookup.
type Policy string

const (
	PolicyEcho   Policy = "echo"
	PolicyMirror Policy = "mirror"
	PolicySilent Policy = "silent"
)

// Stub is one configured stub-agent transport.
type Stub struct {
	AgentID string
	Policy  Policy
	Pubkey  ed25519.PublicKey
	Privkey ed25519.PrivateKey
	LS      *ledger.LedgerSet
}

// New constructs a Stub.
func New(agentID string, policy Policy, pub ed25519.PublicKey, priv ed25519.PrivateKey, ls *ledger.LedgerSet) *Stub {
	return &Stub{AgentID: agentID, Policy: policy, Pubkey: pub, Privkey: priv, LS: ls}
}

// Tick implements scheduler.AgentTransport.
func (s *Stub) Tick(_ context.Context, v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	tool := s.pickTool(v.GetAvailableTools())
	switch tool {
	case "speak":
		return s.speak(v)
	case "vote":
		return s.vote(v)
	default:
		reason := "stub_no_tool"
		if s.Policy == PolicySilent {
			reason = "stub_silent"
		}
		return s.yieldAt(v.GetTick(), reason)
	}
}

// pickTool picks a tool based on policy + availability. echo and mirror
// prefer speak then vote then yield. silent always yields.
func (s *Stub) pickTool(available []string) string {
	if s.Policy == PolicySilent {
		return "yield"
	}
	if containsString(available, "speak") {
		return "speak"
	}
	if containsString(available, "vote") {
		return "vote"
	}
	return "yield"
}

func (s *Stub) speak(v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	var content string
	switch s.Policy {
	case PolicyEcho:
		content = fmt.Sprintf("tick=%d agent=%s", v.GetTick(), s.AgentID)
	case PolicyMirror:
		prev := lastPublicMessage(v)
		if prev == "" {
			content = "re: (none)"
		} else {
			content = "re: " + prev
		}
	default:
		content = fmt.Sprintf("stub=%s tick=%d", s.Policy, v.GetTick())
	}
	payload, err := crypto.CanonicalBytes(&eventsv1.Speak{ChannelId: "public", Content: content})
	if err != nil {
		return nil, err
	}
	return s.signedEnvelope(v.GetTick(), "Speak", payload, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
}

func (s *Stub) vote(v *viewv1.View) (*eventsv1.EventEnvelope, error) {
	// Deterministic vote: echo always "approve", mirror always "reject",
	// silent never reaches here. Different by-policy votes give
	// distinguishable outcomes in experiment manifests.
	option := "approve"
	if s.Policy == PolicyMirror {
		option = "reject"
	}
	payload, err := crypto.CanonicalBytes(&eventsv1.Vote{Option: option})
	if err != nil {
		return nil, err
	}
	return s.signedEnvelope(v.GetTick(), "Vote", payload, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
}

func (s *Stub) yieldAt(tick uint64, reason string) (*eventsv1.EventEnvelope, error) {
	payload, err := crypto.CanonicalBytes(&eventsv1.Yield{Reason: reason})
	if err != nil {
		return nil, err
	}
	return s.signedEnvelope(tick, "Yield", payload, ledger.Destination{LedgerID: ledgerv1.LedgerID_LEDGER_ID_L_PUB})
}

func (s *Stub) signedEnvelope(tick uint64, eventType string, payload []byte, dest ledger.Destination) (*eventsv1.EventEnvelope, error) {
	env := &eventsv1.EventEnvelope{
		EmitterPubkey:     s.Pubkey,
		Tick:              tick,
		SequencePerLedger: s.LS.NextSeqFor(s.Pubkey, dest),
		EventType:         eventType,
		Payload:           payload,
	}
	sig, err := crypto.SignEnvelope(s.Privkey, env)
	if err != nil {
		return nil, err
	}
	env.Signature = sig
	return env, nil
}

// lastPublicMessage returns the content of the last Speak on `public`,
// or "" if none.
func lastPublicMessage(v *viewv1.View) string {
	for _, ch := range v.GetChannelHistories() {
		if ch.GetChannelId() != "public" {
			continue
		}
		msgs := ch.GetMessages()
		if len(msgs) == 0 {
			return ""
		}
		last := msgs[len(msgs)-1].GetContent()
		// Strip any "re: " mirror prefix to avoid pathological growth
		// across rounds where mirror echoes its own previous output.
		return strings.TrimPrefix(last, "re: ")
	}
	return ""
}

func containsString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// Lookup turns a YAML stub_policy string into a Policy. Unknown values
// default to silent (so a typo doesn't silently produce echo behavior).
func Lookup(s string) Policy {
	switch Policy(s) {
	case PolicyEcho:
		return PolicyEcho
	case PolicyMirror:
		return PolicyMirror
	case PolicySilent, "":
		return PolicySilent
	default:
		return PolicySilent
	}
}
