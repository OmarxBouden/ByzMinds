// Package handler owns the kernel's world state (agent roster, channel
// memberships, capability tables, scenario phase tracker, external-
// injection queue) and exposes the ten control methods that mutate it.
//
// Each control method:
//   1. Computes effective_tick from the scheduler's tick view.
//   2. Writes a kernel-signed HandlerControlEvent to L_ctrl recording
//      the RPC name + canonical request bytes + effective_tick.
//   3. Either mutates state immediately or queues a deferred mutation
//      keyed by effective_tick (the scheduler drains the queue at the
//      start of each tick, before step 4's L_cog_ind snapshot).
//   4. Returns a HandlerAck pinning the L_ctrl commit metadata.
//
// Read accessors used by the view builder and scheduler live alongside
// the mutators; they take the lock only briefly to copy the values they
// need so callers never iterate over live state.
package handler

import (
	"sort"
	"sync"
)

// AgentState is the kernel-known facts about one spawned agent.
type AgentState struct {
	ID         string
	Pubkey     []byte
	Role       string
	StubPolicy string
	Theta      []float64 // length 6 (six dials); zero in Stage A baselines
	SpawnTick  uint64
	KilledAt   uint64 // 0 means still alive
	alive      bool
}

// ChannelState records a private channel's roster and lifecycle.
type ChannelState struct {
	ChannelID  string
	Members    []string // agent_ids, sorted alphabetically
	OpenedAt   uint64
	ClosedAt   uint64 // 0 means still open
}

// CapabilityState holds the load state for one agent's capabilities.
// Capabilities are flat strings (no separate registry in Step 2). A
// capability is loaded ↔ present in Loaded.
type CapabilityState struct {
	Loaded map[string]bool
}

// PhaseSpec describes one phase of the scenario timeline.
type PhaseSpec struct {
	Name           string
	Rounds         uint32
	AvailableTools []string // sorted alphabetically for canonical view rendering
}

// ScenarioState is the kernel-side snapshot of the loaded scenario YAML.
type ScenarioState struct {
	Name           string
	YAMLHash       string // hex-encoded SHA-256 of the YAML bytes
	TaskArtifact   string
	Phases         []PhaseSpec
	HistoryWindow  map[string]uint32 // channel_id → K_c; "public" + "private" defaults
}

// PendingExternal is one queued InjectExternalMessage waiting to be
// surfaced in the next BuildView for its target agent.
type PendingExternal struct {
	ClaimedSource string
	Content       string
	InjectTick    uint64
}

// SchedulerControl is the narrow interface the handler uses to drive
// the scheduler (Pause/Resume/Step) and read the current tick. The
// scheduler package implements this; the handler holds it via
// SetScheduler after construction to break the import cycle.
type SchedulerControl interface {
	CurrentTick() uint64
	Pause()
	Resume()
	Step(n uint32)
}

// state is the lock-protected world state. All fields are mutated
// exclusively through handler methods.
type state struct {
	mu sync.Mutex

	// Agent registry.
	agents       map[string]*AgentState // agent_id → state
	pubkeyToID   map[string]string      // hex(pubkey) → agent_id
	liveOrdering []string               // sorted agent_id list (alive ones), kept in sync

	// Channels.
	channels map[string]*ChannelState

	// Per-agent capability load state.
	caps map[string]*CapabilityState

	// Scenario.
	scenario *ScenarioState

	// Per-agent pending external messages (FIFO arrival order).
	externals map[string][]PendingExternal

	// Per-agent latest task assignment (single, overwritten on AssignTask).
	taskAssignments map[string][]byte
	taskKinds       map[string]string

	// Deferred mutations to apply at the *start* of a future tick,
	// before that tick's step 4 (L_cog_ind snapshot). Used for Retune,
	// agent-initiated OpenChannel approval, and KillAgent.
	queuedAt map[uint64][]queuedOp
}

type queuedOp struct {
	label string
	apply func() error
}

func newState() *state {
	return &state{
		agents:          make(map[string]*AgentState),
		pubkeyToID:      make(map[string]string),
		channels:        make(map[string]*ChannelState),
		caps:            make(map[string]*CapabilityState),
		externals:       make(map[string][]PendingExternal),
		taskAssignments: make(map[string][]byte),
		taskKinds:       make(map[string]string),
		queuedAt:        make(map[uint64][]queuedOp),
	}
}

// rebuildLiveOrderingLocked recomputes the sorted alive-agent list. Call
// after any change to agents[id].alive. Held: state.mu.
func (s *state) rebuildLiveOrderingLocked() {
	out := make([]string, 0, len(s.agents))
	for id, a := range s.agents {
		if a.alive {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	s.liveOrdering = out
}
