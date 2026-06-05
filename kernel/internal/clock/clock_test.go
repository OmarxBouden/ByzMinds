package clock

import (
	"sync"
	"testing"
)

func TestTickAdvanceAndSet(t *testing.T) {
	tk := NewTick(0)
	if got := tk.Now(); got != 0 {
		t.Fatalf("initial tick = %d, want 0", got)
	}
	if got := tk.Advance(); got != 1 {
		t.Fatalf("Advance() = %d, want 1", got)
	}
	tk.Set(100)
	if got := tk.Now(); got != 100 {
		t.Fatalf("Set then Now = %d, want 100", got)
	}
}

func TestSequenceMonotonicAndConcurrent(t *testing.T) {
	s := NewSequence(0)
	const goroutines = 16
	const perG = 1000
	var wg sync.WaitGroup
	results := make(chan uint64, goroutines*perG)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < perG; j++ {
				results <- s.Next()
			}
		}()
	}
	wg.Wait()
	close(results)

	seen := make(map[uint64]bool, goroutines*perG)
	for r := range results {
		if seen[r] {
			t.Fatalf("duplicate sequence %d", r)
		}
		seen[r] = true
	}
	if len(seen) != goroutines*perG {
		t.Fatalf("expected %d unique seqs, got %d", goroutines*perG, len(seen))
	}
	if got := s.Last(); got != uint64(goroutines*perG) {
		t.Fatalf("Last() = %d, want %d", got, goroutines*perG)
	}
}

func TestOrderLess(t *testing.T) {
	cases := []struct {
		a, b Order
		want bool
	}{
		{Order{1, 1, 1}, Order{2, 1, 1}, true},  // earlier tick wins
		{Order{2, 1, 1}, Order{1, 1, 1}, false}, // later tick loses
		{Order{1, 1, 1}, Order{1, 2, 1}, true},  // earlier ledger wins on same tick
		{Order{1, 1, 1}, Order{1, 1, 2}, true},  // earlier seq wins on same (tick, ledger)
		{Order{1, 1, 5}, Order{1, 1, 5}, false}, // equal is not less
	}
	for _, c := range cases {
		if got := c.a.Less(c.b); got != c.want {
			t.Errorf("Less(%+v, %+v) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestSequenceTableLazyCreate(t *testing.T) {
	st := NewSequenceTable()
	a := st.GetOrCreate("L_pub")
	b := st.GetOrCreate("L_pub")
	if a != b {
		t.Fatal("GetOrCreate must return the same Sequence for the same key")
	}
	a.Next()
	a.Next()
	if got := b.Last(); got != 2 {
		t.Fatalf("aliased Sequence.Last = %d, want 2", got)
	}

	other := st.GetOrCreate("L_prv:ch_07")
	if other == a {
		t.Fatal("different keys must yield different Sequences")
	}
}

func TestSequenceTableSnapshot(t *testing.T) {
	st := NewSequenceTable()
	st.GetOrCreate("a").Next()
	st.GetOrCreate("a").Next()
	st.GetOrCreate("b").Next()
	st.GetOrCreate("c") // never advanced
	snap := st.Snapshot()
	if snap["a"] != 2 || snap["b"] != 1 || snap["c"] != 0 {
		t.Fatalf("snapshot mismatch: %+v", snap)
	}
}
