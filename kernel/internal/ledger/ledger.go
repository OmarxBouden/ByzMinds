// Package ledger holds the kernel's append-only logs.
//
// One LedgerSet owns the five concrete ledgers and serializes all commits
// behind a single mutex. The serialization gives us:
//
//   - Per-(emitter, destination) sequence numbering. Each (emitter_pubkey,
//     ledger_id, channel_id) tuple has its own monotonic counter; the kernel
//     rejects any envelope whose sequence_per_ledger does not equal that
//     counter's next value. Two distinct emitters can therefore both claim
//     seq=1 on L_pub without colliding — concurrent submission "just works"
//     without requiring the harness to coordinate sequence allocation.
//   - A linear hash chain over the merged commit log: every committed event
//     carries prev_chain_hash + chain_hash, where chain_hash is
//     SHA-256(canonical(ChainInput)) and ChainInput.prev_chain_hash points
//     to the previous CommittedEvent in global_commit_seq order. The chain
//     is global (not per-ledger) so that the manifest's last chain_hash is
//     a single integrity commitment over the whole run. This is unaffected
//     by the per-(emitter, ledger) seq scheme.
//
// Identities are Ed25519 public keys. Access policy is read-only here —
// write admissibility (phases, channel membership, cap state) lives in
// the schema package.
package ledger

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"sync"

	"github.com/byzminds/byzminds/kernel/internal/clock"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// ErrSequenceMismatch is returned when envelope.sequence_per_ledger is
// not the next seq for the (emitter, destination) pair.
var ErrSequenceMismatch = errors.New("ledger: sequence_per_ledger does not match next_seq for emitter")

// ErrUnknownChannel is returned when a commit targets L_prv with a
// channel_id that has not been opened.
var ErrUnknownChannel = errors.New("ledger: unknown private channel")

// ErrInvalidLedger is returned when the destination LedgerID is not one
// of the five configured ledgers.
var ErrInvalidLedger = errors.New("ledger: invalid destination")

// CommitTime is the wall-clock source used for diagnostic
// CommittedEvent.commit_unix_nanos. Held as a function so tests can pin it.
type CommitTime func() uint64

// Ledger is one append-only log. It holds access policy and the ordered
// events. Sequence numbering is per-(emitter, ledger) and lives on the
// owning LedgerSet, not here. The LedgerSet's mutex serializes mutations;
// the per-ledger RWMutex guards reader access to the events slice.
type Ledger struct {
	id        ledgerv1.LedgerID
	channelID string // empty unless id == L_PRV
	name      string // human-readable: "L_pub", "L_prv:ch_07", etc.
	access    AccessPolicy
	frozen    bool // L_PRV channels that have been closed; reject writes, keep reads

	mu     sync.RWMutex
	events []*ledgerv1.CommittedEvent
}

// ID returns the ledger's enum id.
func (l *Ledger) ID() ledgerv1.LedgerID { return l.id }

// ChannelID returns the channel id (empty unless id == L_PRV).
func (l *Ledger) ChannelID() string { return l.channelID }

// Name returns a human-readable label for logs and tests.
func (l *Ledger) Name() string { return l.name }

// Len returns the number of committed events.
func (l *Ledger) Len() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.events)
}

// CanRead reports whether reader may read this ledger.
func (l *Ledger) CanRead(reader Identity) bool { return l.access.CanRead(reader) }

// Snapshot returns a defensive copy of the events slice ordered by seq.
// Callers wanting a stream should use Iterate.
func (l *Ledger) Snapshot() []*ledgerv1.CommittedEvent {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]*ledgerv1.CommittedEvent, len(l.events))
	copy(out, l.events)
	return out
}

// Iterate returns a channel that yields every event in append order whose
// global_commit_seq >= fromGlobal, then closes. The current implementation
// is a snapshot iterator: it does not block for future appends. Live
// tailing arrives in Step 2.
//
// Note: filtering on global_commit_seq (rather than per-emitter
// sequence_per_ledger) gives an unambiguous "events after this point in
// the run" semantics now that per-ledger seqs are not globally unique.
func (l *Ledger) Iterate(fromGlobal uint64) <-chan *ledgerv1.CommittedEvent {
	l.mu.RLock()
	out := make(chan *ledgerv1.CommittedEvent, len(l.events))
	for _, e := range l.events {
		if e.GetGlobalCommitSeq() >= fromGlobal {
			out <- e
		}
	}
	l.mu.RUnlock()
	close(out)
	return out
}

// LedgerSet owns the five Stage A ledgers and the global commit pipeline.
type LedgerSet struct {
	pub    *Ledger
	prv    map[string]*Ledger
	cogInd *Ledger
	cogEli *Ledger
	ctrl   *Ledger

	researcher Identity
	kernelPriv ed25519.PrivateKey

	mu        sync.Mutex // commit lock; serializes all writes across all ledgers
	globalSeq *clock.Sequence
	chainHead []byte // most recent chain_hash; 32 zero bytes before any commit
	committed []*ledgerv1.CommittedEvent

	// emitterSeq tracks the most-recently-committed sequence per
	// (destination, emitter) pair. Key built by emitterSeqKey. Mutated
	// only under mu (the commit lock).
	emitterSeq map[string]uint64

	commitTime CommitTime
}

// Config configures a fresh LedgerSet.
type Config struct {
	Researcher Identity
	KernelPriv ed25519.PrivateKey
	// PrivateChannels is the set of L_prv[c] ledgers to open at construction.
	// Step 2 will swap this for handler-driven OpenChannel.
	PrivateChannels []PrivateChannelConfig
	// CommitTime is the diagnostic clock for CommittedEvent.commit_unix_nanos.
	// nil ⇒ time.Now().UnixNano().
	CommitTime CommitTime
}

// PrivateChannelConfig opens an L_prv[id] ledger with the given members.
type PrivateChannelConfig struct {
	ChannelID string
	Members   []Identity
}

// New constructs a LedgerSet with the canonical five ledgers wired.
func New(cfg Config) (*LedgerSet, error) {
	if len(cfg.Researcher) != crypto.PublicKeySize {
		return nil, fmt.Errorf("ledger: researcher pubkey must be %d bytes, got %d", crypto.PublicKeySize, len(cfg.Researcher))
	}
	if len(cfg.KernelPriv) != crypto.PrivateKeySize {
		return nil, fmt.Errorf("ledger: kernel private key must be %d bytes, got %d", crypto.PrivateKeySize, len(cfg.KernelPriv))
	}
	ct := cfg.CommitTime
	if ct == nil {
		ct = defaultCommitTime
	}

	ls := &LedgerSet{
		researcher: cfg.Researcher,
		kernelPriv: cfg.KernelPriv,
		globalSeq:  clock.NewSequence(0),
		chainHead:  make([]byte, sha256.Size), // genesis: 32 zero bytes
		commitTime: ct,
		prv:        make(map[string]*Ledger),
		emitterSeq: make(map[string]uint64),
	}

	ls.pub = &Ledger{
		id: ledgerv1.LedgerID_LEDGER_ID_L_PUB, name: "L_pub",
		access: PublicAccess(),
	}
	ls.cogInd = &Ledger{
		id: ledgerv1.LedgerID_LEDGER_ID_L_COG_IND, name: "L_cog_ind",
		access: ResearcherOnly(cfg.Researcher),
	}
	ls.cogEli = &Ledger{
		id: ledgerv1.LedgerID_LEDGER_ID_L_COG_ELI, name: "L_cog_eli",
		access: ResearcherOnly(cfg.Researcher),
	}
	ls.ctrl = &Ledger{
		id: ledgerv1.LedgerID_LEDGER_ID_L_CTRL, name: "L_ctrl",
		access: ResearcherOnly(cfg.Researcher),
	}
	for _, ch := range cfg.PrivateChannels {
		if ch.ChannelID == "" {
			return nil, errors.New("ledger: private channel id must not be empty")
		}
		if _, dup := ls.prv[ch.ChannelID]; dup {
			return nil, fmt.Errorf("ledger: duplicate private channel %q", ch.ChannelID)
		}
		ls.prv[ch.ChannelID] = &Ledger{
			id: ledgerv1.LedgerID_LEDGER_ID_L_PRV, channelID: ch.ChannelID,
			name:   "L_prv:" + ch.ChannelID,
			access: PrivateChannelAccess(cfg.Researcher, ch.Members),
		}
	}
	return ls, nil
}

// emitterSeqKey builds the map key for the per-(emitter, destination)
// counter. Stable across runs because hex encoding is deterministic.
func emitterSeqKey(dest Destination, emitter []byte) string {
	return strconv.Itoa(int(dest.LedgerID)) + "|" + dest.ChannelID + "|" + hex.EncodeToString(emitter)
}

// Researcher returns the researcher pubkey configured at construction.
func (s *LedgerSet) Researcher() Identity { return s.researcher }

// KernelPriv returns the kernel's private key. Used by the scheduler
// and handler to sign synthetic envelopes (CogIndSnapshot, handler
// control events, tick-timeout yields).
func (s *LedgerSet) KernelPriv() ed25519.PrivateKey { return s.kernelPriv }

// NextSeqFor returns the next sequence_per_ledger that an event from
// emitter on dest must claim to commit. Used by the scheduler to mint
// kernel-synthesized envelopes that occupy the agent's would-be slot.
func (s *LedgerSet) NextSeqFor(emitter []byte, dest Destination) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.emitterSeq[emitterSeqKey(dest, emitter)] + 1
}

// OpenPrivateChannel mints a new L_prv[channelID] ledger with members
// as readers. Returns ErrInvalidLedger if a ledger by that id is already
// open. Called by the handler after a successful OpenChannel control
// event commits to L_ctrl.
func (s *LedgerSet) OpenPrivateChannel(channelID string, members []Identity) error {
	if channelID == "" {
		return errors.New("ledger: OpenPrivateChannel requires channel_id")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.prv[channelID]; dup {
		return fmt.Errorf("ledger: channel %q already open", channelID)
	}
	s.prv[channelID] = &Ledger{
		id:        ledgerv1.LedgerID_LEDGER_ID_L_PRV,
		channelID: channelID,
		name:      "L_prv:" + channelID,
		access:    PrivateChannelAccess(s.researcher, members),
	}
	return nil
}

// FreezePrivateChannel marks the channel closed for writes. Reads
// remain available. Implemented by swapping the channel's access
// policy to "readers continue, writers blocked" — Step 2's commit
// pipeline does not consult access policy on writes, so freezing is
// today equivalent to "delete from the writable map and refuse new
// commits to it"; we keep the ledger entry so prior reads still work.
//
// For Step 2 minimal scope we mark a frozen flag inline on the Ledger
// and reject Commit destinations whose target is frozen.
func (s *LedgerSet) FreezePrivateChannel(channelID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.prv[channelID]
	if !ok {
		return fmt.Errorf("ledger: unknown private channel %q", channelID)
	}
	l.frozen = true
	return nil
}

// Pub returns the L_pub ledger.
func (s *LedgerSet) Pub() *Ledger { return s.pub }

// Prv returns the L_prv[channelID] ledger, or nil if none open.
func (s *LedgerSet) Prv(channelID string) *Ledger { return s.prv[channelID] }

// CogInd returns the L_cog_ind ledger.
func (s *LedgerSet) CogInd() *Ledger { return s.cogInd }

// CogEli returns the L_cog_eli ledger.
func (s *LedgerSet) CogEli() *Ledger { return s.cogEli }

// Ctrl returns the L_ctrl ledger.
func (s *LedgerSet) Ctrl() *Ledger { return s.ctrl }

// ChainHead returns a copy of the current chain head (32 bytes).
func (s *LedgerSet) ChainHead() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]byte, len(s.chainHead))
	copy(out, s.chainHead)
	return out
}

// CommittedLog returns a defensive copy of the merged commit log in
// global_commit_seq order. Used by the manifest writer and replay tests.
func (s *LedgerSet) CommittedLog() []*ledgerv1.CommittedEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*ledgerv1.CommittedEvent, len(s.committed))
	copy(out, s.committed)
	return out
}

// Destination identifies one ledger (with channel id for L_prv). The
// schema package resolves an envelope to a Destination before commit.
type Destination struct {
	LedgerID  ledgerv1.LedgerID
	ChannelID string // empty unless LedgerID == L_PRV
}

// Commit appends env to the destination ledger if its envelope.sequence_per_ledger
// matches the destination's next_seq. Returns the resulting CommittedEvent.
//
// All checks that depend on "what's already committed" run under the
// commit lock, so Commit is the kernel's serialization point.
func (s *LedgerSet) Commit(env *eventsv1.EventEnvelope, dest Destination) (*ledgerv1.CommittedEvent, error) {
	if env == nil {
		return nil, errors.New("ledger: nil envelope")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	target, err := s.resolveLocked(dest)
	if err != nil {
		return nil, err
	}
	if target.frozen {
		return nil, fmt.Errorf("ledger: %s is frozen (channel closed)", target.name)
	}

	emKey := emitterSeqKey(dest, env.GetEmitterPubkey())
	expected := s.emitterSeq[emKey] + 1
	if env.GetSequencePerLedger() != expected {
		return nil, fmt.Errorf("%w: ledger=%s emitter=%x expected=%d got=%d",
			ErrSequenceMismatch, target.name, env.GetEmitterPubkey(), expected, env.GetSequencePerLedger())
	}

	prev := make([]byte, len(s.chainHead))
	copy(prev, s.chainHead)
	globalSeq := s.globalSeq.Next()

	chainInput := &ledgerv1.ChainInput{
		Envelope:        env,
		LedgerId:        dest.LedgerID,
		LedgerChannelId: dest.ChannelID,
		GlobalCommitSeq: globalSeq,
		PrevChainHash:   prev,
	}
	chainBytes, err := crypto.CanonicalBytes(chainInput)
	if err != nil {
		return nil, fmt.Errorf("ledger: canonicalize chain input: %w", err)
	}
	sum := sha256.Sum256(chainBytes)
	chainHash := sum[:]
	kernelSig, err := crypto.SignBytes(s.kernelPriv, chainHash)
	if err != nil {
		return nil, fmt.Errorf("ledger: sign chain hash: %w", err)
	}

	committed := &ledgerv1.CommittedEvent{
		Envelope:        env,
		LedgerId:        dest.LedgerID,
		LedgerChannelId: dest.ChannelID,
		GlobalCommitSeq: globalSeq,
		CommitUnixNanos: s.commitTime(),
		PrevChainHash:   prev,
		ChainHash:       chainHash,
		KernelSignature: kernelSig,
	}

	target.mu.Lock()
	target.events = append(target.events, committed)
	target.mu.Unlock()
	s.emitterSeq[emKey] = env.GetSequencePerLedger()

	s.committed = append(s.committed, committed)
	s.chainHead = chainHash

	return committed, nil
}

// AppendReplay re-applies a previously committed event during replay.
// Distinct from Commit because replay must re-derive chain_hash and
// kernel_signature from the manifest's stored values rather than
// re-computing them — this is what lets the replay catch divergence.
//
// Returns an error if any of the rebuild checks fail.
func (s *LedgerSet) AppendReplay(committed *ledgerv1.CommittedEvent) error {
	if committed == nil {
		return errors.New("ledger: nil committed event")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	dest := Destination{LedgerID: committed.GetLedgerId(), ChannelID: committed.GetLedgerChannelId()}
	target, err := s.resolveLocked(dest)
	if err != nil {
		return err
	}
	emKey := emitterSeqKey(dest, committed.GetEnvelope().GetEmitterPubkey())
	if want := s.emitterSeq[emKey] + 1; committed.GetEnvelope().GetSequencePerLedger() != want {
		return fmt.Errorf("%w: ledger=%s emitter=%x expected=%d got=%d",
			ErrSequenceMismatch, target.name, committed.GetEnvelope().GetEmitterPubkey(), want, committed.GetEnvelope().GetSequencePerLedger())
	}
	expectedGlobal := s.globalSeq.Last() + 1
	if committed.GetGlobalCommitSeq() != expectedGlobal {
		return fmt.Errorf("ledger: global_commit_seq mismatch on replay: expected=%d got=%d", expectedGlobal, committed.GetGlobalCommitSeq())
	}
	// The replayed event must extend the current chain head.
	if !bytesEqual(committed.GetPrevChainHash(), s.chainHead) {
		return errors.New("ledger: replay prev_chain_hash does not extend current chain head")
	}
	// Recompute chain_hash and compare. This is what catches divergence.
	chainInput := &ledgerv1.ChainInput{
		Envelope:        committed.GetEnvelope(),
		LedgerId:        committed.GetLedgerId(),
		LedgerChannelId: committed.GetLedgerChannelId(),
		GlobalCommitSeq: committed.GetGlobalCommitSeq(),
		PrevChainHash:   committed.GetPrevChainHash(),
	}
	chainBytes, err := crypto.CanonicalBytes(chainInput)
	if err != nil {
		return fmt.Errorf("ledger: canonicalize chain input on replay: %w", err)
	}
	sum := sha256.Sum256(chainBytes)
	if !bytesEqual(sum[:], committed.GetChainHash()) {
		return errors.New("ledger: replay chain_hash mismatch")
	}

	target.mu.Lock()
	target.events = append(target.events, committed)
	target.mu.Unlock()
	s.emitterSeq[emKey] = committed.GetEnvelope().GetSequencePerLedger()
	s.globalSeq.Next()
	s.committed = append(s.committed, committed)
	s.chainHead = committed.GetChainHash()
	return nil
}

func (s *LedgerSet) resolveLocked(dest Destination) (*Ledger, error) {
	switch dest.LedgerID {
	case ledgerv1.LedgerID_LEDGER_ID_L_PUB:
		if dest.ChannelID != "" {
			return nil, errors.New("ledger: L_PUB must have empty channel id")
		}
		return s.pub, nil
	case ledgerv1.LedgerID_LEDGER_ID_L_PRV:
		if dest.ChannelID == "" {
			return nil, errors.New("ledger: L_PRV requires non-empty channel id")
		}
		l, ok := s.prv[dest.ChannelID]
		if !ok {
			return nil, fmt.Errorf("%w: %q", ErrUnknownChannel, dest.ChannelID)
		}
		return l, nil
	case ledgerv1.LedgerID_LEDGER_ID_L_COG_IND:
		return s.cogInd, nil
	case ledgerv1.LedgerID_LEDGER_ID_L_COG_ELI:
		return s.cogEli, nil
	case ledgerv1.LedgerID_LEDGER_ID_L_CTRL:
		return s.ctrl, nil
	default:
		return nil, fmt.Errorf("%w: %v", ErrInvalidLedger, dest.LedgerID)
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
