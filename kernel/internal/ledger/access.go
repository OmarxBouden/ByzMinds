package ledger

import (
	"bytes"
	"crypto/ed25519"
)

// Identity is whoever is asking to read or write — typically an Ed25519
// public key. The zero value (nil/empty slice) is invalid.
type Identity = ed25519.PublicKey

// equalIdentity returns true iff a and b are the same key bytes.
func equalIdentity(a, b Identity) bool {
	return len(a) > 0 && bytes.Equal(a, b)
}

// AccessPolicy decides whether a given identity may read a ledger. Per
// the design: L_pub is broadly readable, L_prv is members-only, and
// L_cog_ind / L_cog_eli / L_ctrl are researcher-only.
//
// Write admissibility is *not* covered here — it lives in the schema
// package (which knows about phases, channel membership, capability
// state, etc.). access.go is read-side only.
type AccessPolicy interface {
	CanRead(reader Identity) bool
}

// public ledger — readable by everyone.
type publicAccess struct{}

func (publicAccess) CanRead(_ Identity) bool { return true }

// researcherOnly — only the kernel's designated researcher pubkey reads.
type researcherOnly struct{ researcher Identity }

func (r researcherOnly) CanRead(reader Identity) bool {
	return equalIdentity(r.researcher, reader)
}

// privateChannel — members only, plus researcher.
type privateChannel struct {
	researcher Identity
	members    []Identity
}

func (p privateChannel) CanRead(reader Identity) bool {
	if equalIdentity(p.researcher, reader) {
		return true
	}
	for _, m := range p.members {
		if equalIdentity(m, reader) {
			return true
		}
	}
	return false
}

// PublicAccess returns the policy used by L_pub.
func PublicAccess() AccessPolicy { return publicAccess{} }

// ResearcherOnly returns the policy used by L_cog_ind, L_cog_eli, L_ctrl.
func ResearcherOnly(researcher Identity) AccessPolicy {
	return researcherOnly{researcher: researcher}
}

// PrivateChannelAccess returns the policy used by an L_prv[channel].
// Membership is fixed at construction in Step 1; Step 2 will rewire this
// when OpenChannelReq / CloseChannelReq become admissible.
func PrivateChannelAccess(researcher Identity, members []Identity) AccessPolicy {
	cp := make([]Identity, len(members))
	copy(cp, members)
	return privateChannel{researcher: researcher, members: cp}
}
