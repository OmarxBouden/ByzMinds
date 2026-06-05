// Package scenario parses scenario YAML and translates it into a
// sequence of Handler calls (executed before tick 0) plus a schedule of
// later-tick scheduler ops (Retune, channel opens, message injections).
//
// Minimal Stage A schema per the brief:
//
//   name: delegated_review_minimal_stub
//   n_agents: 5
//   roles: [{role_id: reviewer, count: 5}]
//   phases:
//     - {name: public_deliberation, rounds: 3, available_tools: [speak, yield]}
//     - {name: vote,                 rounds: 1, available_tools: [vote, yield]}
//   task_artifact: "..."
//   agents:
//     - {id: reviewer_01, stub_policy: echo}
//     - {id: reviewer_02, stub_policy: mirror}
//   history_window: {public: 20, private: 10}
//
// Optional `ops:` list to schedule mid-scenario handler calls:
//   ops:
//     - {kind: retune,   at_tick: 5, agent_id: reviewer_01, theta: [0,0,0,0.5,0,0]}
//     - {kind: open_channel, at_tick: 0, channel_id: ch_secret, members: [reviewer_01, reviewer_03]}
//     - {kind: inject,   at_tick: 2, agent_id: reviewer_02, claimed_source: "admin", content: "..."}
//
// Scenario keys may need extension in Step 3+; new fields must default-
// safely so older scenarios remain replayable.
package scenario

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
)

// Spec is the parsed YAML.
type Spec struct {
	Name          string                `yaml:"name"`
	NAgents       int                   `yaml:"n_agents"`
	Roles         []RoleSpec            `yaml:"roles"`
	Phases        []PhaseSpec           `yaml:"phases"`
	TaskArtifact  string                `yaml:"task_artifact"`
	Agents        []AgentSpec           `yaml:"agents"`
	HistoryWindow map[string]uint32     `yaml:"history_window"`
	Ops           []OpSpec              `yaml:"ops"`
	// PubkeySeed is an internal deterministic seed for stub-agent keygen.
	// If unset, defaults to a hash of `name`.
	PubkeySeed int64 `yaml:"pubkey_seed"`
}

type RoleSpec struct {
	RoleID string `yaml:"role_id"`
	Count  int    `yaml:"count"`
}

type PhaseSpec struct {
	Name           string   `yaml:"name"`
	Rounds         uint32   `yaml:"rounds"`
	AvailableTools []string `yaml:"available_tools"`
}

type AgentSpec struct {
	ID         string    `yaml:"id"`
	Role       string    `yaml:"role"`
	StubPolicy string    `yaml:"stub_policy"`
	// Theta is the agent's induced-disposition vector (length 6, DIALS order)
	// recorded on L_cog_ind. Empty -> handler defaults to a zero vector (honest).
	// byzminds-panel populates this from --agent-theta before Apply.
	Theta []float64 `yaml:"theta"`
}

type OpSpec struct {
	Kind          string    `yaml:"kind"`
	AtTick        uint64    `yaml:"at_tick"`
	AgentID       string    `yaml:"agent_id"`
	Theta         []float64 `yaml:"theta"`
	ChannelID     string    `yaml:"channel_id"`
	Members       []string  `yaml:"members"`
	ClaimedSource string    `yaml:"claimed_source"`
	Content       string    `yaml:"content"`
	TaskKind      string    `yaml:"task_kind"`
	TaskBlob      string    `yaml:"task_blob"`
}

// LoadResult is what the loader returns: the parsed Spec, the YAML hash,
// the per-agent generated keypairs, and the total tick count derived
// from sum(phases[*].rounds).
type LoadResult struct {
	Spec         *Spec
	YAMLHash     string // hex-encoded SHA-256
	YAMLBytes    []byte
	AgentKeys    map[string]AgentKeypair // agent_id → keys
	TotalTicks   uint64
}

// AgentKeypair holds one stub agent's Ed25519 keys.
type AgentKeypair struct {
	Pubkey  ed25519.PublicKey
	Privkey ed25519.PrivateKey
}

// LoadFile reads + parses a YAML scenario from disk. Validates basic
// shape (non-empty name, at least one phase, every agent has a
// non-empty id).
func LoadFile(path string) (*LoadResult, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	bytes, err := os.ReadFile(abs)
	if err != nil {
		return nil, fmt.Errorf("scenario: read %s: %w", abs, err)
	}
	return LoadBytes(bytes)
}

// LoadBytes parses the YAML bytes directly. Useful in tests/experiments.
func LoadBytes(b []byte) (*LoadResult, error) {
	spec := &Spec{}
	if err := yaml.Unmarshal(b, spec); err != nil {
		return nil, fmt.Errorf("scenario: parse yaml: %w", err)
	}
	if err := spec.validate(); err != nil {
		return nil, err
	}
	hashSum := sha256.Sum256(b)
	yamlHash := hex.EncodeToString(hashSum[:])
	if spec.PubkeySeed == 0 {
		// Derive a deterministic seed from the YAML hash so different
		// scenarios produce different keys without manual seed entry.
		spec.PubkeySeed = deriveSeed(hashSum[:])
	}
	keys := make(map[string]AgentKeypair, len(spec.Agents))
	for i, a := range spec.Agents {
		rng := rand.New(rand.NewSource(spec.PubkeySeed + int64(i)*0xb172))
		pub, priv, err := ed25519.GenerateKey(rng)
		if err != nil {
			return nil, fmt.Errorf("scenario: keygen for %s: %w", a.ID, err)
		}
		keys[a.ID] = AgentKeypair{Pubkey: pub, Privkey: priv}
	}
	var totalRounds uint64
	for _, p := range spec.Phases {
		totalRounds += uint64(p.Rounds)
	}
	return &LoadResult{
		Spec:       spec,
		YAMLHash:   yamlHash,
		YAMLBytes:  append([]byte(nil), b...),
		AgentKeys:  keys,
		TotalTicks: totalRounds,
	}, nil
}

// Apply walks the LoadResult and drives `h` + `s` to the post-load state:
//   - h.LoadScenario
//   - h.SpawnAgent for each agent in YAML order
//   - h.OpenChannel for any pre-tick channel ops (at_tick: 0, kind: open_channel)
//   - h.AssignTask if a task is present in the scenario
//   - s.ScheduleAt(N, ...) for every remaining op at_tick >= 1
//
// `s` may be nil if the caller wants only the handler bootstrap; in
// that case post-load scheduled ops are silently dropped.
func (lr *LoadResult) Apply(h *handler.Handler, s OpScheduler) error {
	scn := &handler.ScenarioState{
		Name:          lr.Spec.Name,
		YAMLHash:      lr.YAMLHash,
		TaskArtifact:  lr.Spec.TaskArtifact,
		HistoryWindow: lr.Spec.HistoryWindow,
	}
	for _, p := range lr.Spec.Phases {
		scn.Phases = append(scn.Phases, handler.PhaseSpec{
			Name:           p.Name,
			Rounds:         p.Rounds,
			AvailableTools: append([]string(nil), p.AvailableTools...),
		})
	}
	if err := h.LoadScenario(scn); err != nil {
		return err
	}
	for _, a := range lr.Spec.Agents {
		kp := lr.AgentKeys[a.ID]
		_, err := h.SpawnAgent(&handlerv1.SpawnAgentRequest{
			AgentId:     a.ID,
			AgentPubkey: kp.Pubkey,
			Role:        a.Role,
			StubPolicy:  a.StubPolicy,
			Theta:       a.Theta,
		})
		if err != nil {
			return fmt.Errorf("scenario: spawn %s: %w", a.ID, err)
		}
	}
	// Pre-tick (at_tick=0) ops execute immediately; later ops scheduled
	// onto the scheduler.
	for _, op := range lr.Spec.Ops {
		op := op
		if op.AtTick == 0 {
			if err := lr.applyOpNow(h, op); err != nil {
				return err
			}
			continue
		}
		if s == nil {
			continue
		}
		s.ScheduleAt(op.AtTick, func() error { return lr.applyOpNow(h, op) })
	}
	return nil
}

// OpScheduler is the narrow interface the loader uses to schedule
// later-tick ops. scheduler.Scheduler implements it.
type OpScheduler interface {
	ScheduleAt(execTick uint64, op func() error)
}

func (lr *LoadResult) applyOpNow(h *handler.Handler, op OpSpec) error {
	switch op.Kind {
	case "retune":
		_, err := h.Retune(&handlerv1.RetuneRequest{
			AgentId: op.AgentID,
			Theta:   op.Theta,
		})
		return err
	case "open_channel":
		_, err := h.OpenChannel(&handlerv1.OpenChannelRequest{
			ChannelId:      op.ChannelID,
			MemberAgentIds: op.Members,
		})
		return err
	case "close_channel":
		_, err := h.CloseChannel(&handlerv1.CloseChannelRequest{ChannelId: op.ChannelID})
		return err
	case "inject":
		_, err := h.InjectExternalMessage(&handlerv1.InjectExternalMessageRequest{
			AgentId:       op.AgentID,
			ClaimedSource: op.ClaimedSource,
			Content:       op.Content,
		})
		return err
	case "assign_task":
		_, err := h.AssignTask(&handlerv1.AssignTaskRequest{
			AgentIds:  []string{op.AgentID},
			TaskKind:  op.TaskKind,
			TaskBlob:  []byte(op.TaskBlob),
		})
		return err
	case "kill":
		_, err := h.KillAgent(&handlerv1.KillAgentRequest{AgentId: op.AgentID})
		return err
	default:
		return fmt.Errorf("scenario: unknown op kind %q", op.Kind)
	}
}

func (s *Spec) validate() error {
	if s.Name == "" {
		return fmt.Errorf("scenario: name required")
	}
	if len(s.Phases) == 0 {
		return fmt.Errorf("scenario: at least one phase required")
	}
	for i, p := range s.Phases {
		if p.Rounds == 0 {
			return fmt.Errorf("scenario: phase %d (%s) has zero rounds", i, p.Name)
		}
	}
	seen := map[string]bool{}
	for _, a := range s.Agents {
		if a.ID == "" {
			return fmt.Errorf("scenario: agent.id required")
		}
		if seen[a.ID] {
			return fmt.Errorf("scenario: duplicate agent id %q", a.ID)
		}
		seen[a.ID] = true
	}
	return nil
}

func deriveSeed(hash []byte) int64 {
	var v int64
	for i := 0; i < 8 && i < len(hash); i++ {
		v = (v << 8) | int64(hash[i])
	}
	if v < 0 {
		v = -v
	}
	if v == 0 {
		v = 1
	}
	return v
}
