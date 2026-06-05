// Package clock holds the kernel's logical clocks. Two distinct concepts
// live here:
//
//   - The global tick counter — one per kernel instance. Advanced by the
//     dispatch loop in Step 2; held as a simple atomic for now.
//   - Per-ledger sequence counters — one per concrete ledger (a per-channel
//     counter for L_prv). Append() returns the next sequence and increments
//     atomically; the ledger package owns the wiring.
//
// We do not use vector clocks: the kernel is a single process and ordering
// is dictated by the kernel's commit lock. Cross-ledger total order is
// (tick, ledger_id, sequence_per_ledger) lex.
package clock

import (
	"sync"
	"sync/atomic"
)

// Tick is a kernel-wide monotonic counter. Zero is the initial state;
// dispatch advances it via Advance().
type Tick struct {
	v atomic.Uint64
}

// NewTick returns a Tick at start (typically 0).
func NewTick(start uint64) *Tick {
	t := &Tick{}
	t.v.Store(start)
	return t
}

// Now returns the current tick.
func (t *Tick) Now() uint64 { return t.v.Load() }

// Advance moves to t+1 and returns the new value. Step 2 wires this into
// the dispatcher; Step 1 callers can advance manually in tests.
func (t *Tick) Advance() uint64 { return t.v.Add(1) }

// Set forces the tick to v. Used only by replay against a saved manifest.
func (t *Tick) Set(v uint64) { t.v.Store(v) }

// Sequence is a monotonic counter for one ledger. Zero means "no events
// committed yet"; the first Next() returns 1.
type Sequence struct {
	v atomic.Uint64
}

// NewSequence returns a Sequence whose first Next() returns start+1.
// Replay starts from the manifest's last seq.
func NewSequence(start uint64) *Sequence {
	s := &Sequence{}
	s.v.Store(start)
	return s
}

// Next allocates and returns the next sequence number atomically.
func (s *Sequence) Next() uint64 { return s.v.Add(1) }

// Last returns the most-recently-allocated sequence (0 if none).
func (s *Sequence) Last() uint64 { return s.v.Load() }

// Order is the total order key over the merged commit log:
//
//	(tick, ledger_id, sequence_per_ledger).
//
// Used by GetView merging and replay verification.
type Order struct {
	Tick     uint64
	LedgerID int32 // matches ledgerv1.LedgerID's underlying type
	Seq      uint64
}

// Less implements lex order for Order.
func (a Order) Less(b Order) bool {
	if a.Tick != b.Tick {
		return a.Tick < b.Tick
	}
	if a.LedgerID != b.LedgerID {
		return a.LedgerID < b.LedgerID
	}
	return a.Seq < b.Seq
}

// SequenceTable is a lazily-populated map of named-sequence → counter.
// The ledger package uses one per ledger family — for instance one entry
// per private channel under L_prv.
type SequenceTable struct {
	mu sync.Mutex
	m  map[string]*Sequence
}

// NewSequenceTable returns an empty table.
func NewSequenceTable() *SequenceTable {
	return &SequenceTable{m: make(map[string]*Sequence)}
}

// GetOrCreate returns the Sequence for key, creating it (starting at 0)
// if absent. Concurrency-safe.
func (t *SequenceTable) GetOrCreate(key string) *Sequence {
	t.mu.Lock()
	defer t.mu.Unlock()
	if s, ok := t.m[key]; ok {
		return s
	}
	s := NewSequence(0)
	t.m[key] = s
	return s
}

// Snapshot returns the current Last() values for every key. Used by the
// manifest writer to record per-ledger sequence heads.
func (t *SequenceTable) Snapshot() map[string]uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]uint64, len(t.m))
	for k, s := range t.m {
		out[k] = s.Last()
	}
	return out
}
