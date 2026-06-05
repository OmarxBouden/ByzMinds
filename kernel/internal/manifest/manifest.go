// Package manifest writes and replays run manifests.
//
// A manifest is a gzipped JSON document that captures everything needed
// to reproduce a run end-to-end:
//
//   - Schema version (so readers can tolerate future field additions).
//   - Kernel binary version + build hash.
//   - Initial ledger state — names and access-policy summary at boot.
//   - Every committed event in global_commit_seq order, encoded with
//     protojson (so unknown fields written by a newer kernel survive a
//     round-trip through an older reader).
//   - The final chain_hash, hex-encoded. Matching this against a fresh
//     replay's chain head is the integrity check.
//
// Replay is the bit-determinism oracle for Step 1's Experiment 003.
package manifest

import (
	"bufio"
	"compress/gzip"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/byzminds/byzminds/kernel/internal/ledger"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
)

// SchemaVersion is the manifest schema. Bump on a breaking change.
// Readers must tolerate fields they do not recognize.
const SchemaVersion uint32 = 1

// Manifest is the on-disk root document.
type Manifest struct {
	SchemaVersion  uint32            `json:"schema_version"`
	KernelVersion  string            `json:"kernel_version"`
	BuildHash      string            `json:"build_hash"`
	InitialState   InitialState      `json:"initial_state"`
	// ModelVersions (Step 5): records the exact model revisions that
	// produced the run so Stage A's reproducibility guarantee is
	// hash-pinned. Mixed-revision aggregation is forbidden per the
	// Step 5 brief — replay against a different revision must fail
	// fast rather than silently shift outputs.
	ModelVersions  ModelVersions     `json:"model_versions,omitempty"`
	Events         []json.RawMessage `json:"events"`
	FinalChainHash string            `json:"final_chain_hash"` // hex
}

// ModelVersions records the model revision SHAs needed to reproduce
// a run. Every field is optional (Step 1-4 manifests don't populate
// it); a Step 5+ run that touches an LLM populates everything.
type ModelVersions struct {
	// AgentModel is the HF model identifier, e.g.
	// "meta-llama/Llama-3.1-8B-Instruct".
	AgentModel string `json:"agent_model,omitempty"`
	// AgentModelRevisionSHA is the HF revision SHA pinning the exact
	// weights snapshot. A vendor re-tag changes this without changing
	// AgentModel, which is the failure mode the Step 5 brief calls
	// out as cache-invalidating.
	AgentModelRevisionSHA string `json:"agent_model_revision_sha,omitempty"`
	// AgentModelDtype is the loaded dtype, e.g. "bf16".
	AgentModelDtype string `json:"agent_model_dtype,omitempty"`
	// VLLMBuildSHA is the vLLM build identifier (commit SHA preferred,
	// else the pip-installed version string).
	VLLMBuildSHA string `json:"vllm_build_sha,omitempty"`
	// JudgeModel + JudgeModelRevisionSHA carry the surface-honesty
	// judge's identity (Stage A: Qwen 2.5 14B Instruct).
	JudgeModel            string `json:"judge_model,omitempty"`
	JudgeModelRevisionSHA string `json:"judge_model_revision_sha,omitempty"`
}

// InitialState describes the kernel's boot configuration. Access policy
// is captured by ledger name + a one-line policy summary; the full policy
// is reconstructible from the ledger names and the channel-members table.
type InitialState struct {
	ResearcherPubkeyHex string             `json:"researcher_pubkey_hex"`
	Ledgers             []LedgerSummary    `json:"ledgers"`
	PrivateChannels     []ChannelSummary   `json:"private_channels"`
}

type LedgerSummary struct {
	Name        string `json:"name"`
	AccessShape string `json:"access_shape"` // "public" | "researcher_only" | "private_channel"
}

type ChannelSummary struct {
	ChannelID    string   `json:"channel_id"`
	MemberPubkeysHex []string `json:"member_pubkeys_hex"`
}

// Header captures kernel-version metadata. Held separately so callers
// can reuse one Header across many runs.
type Header struct {
	KernelVersion string
	BuildHash     string
	// ModelVersions (Step 5) is optional — Step 1/2 runs with no LLM
	// leave it zero-valued and the resulting manifest omits the
	// section entirely (``omitempty`` on the Manifest field).
	ModelVersions ModelVersions
}

// Write serializes ls's commit log into a gzipped manifest on w.
func Write(w io.Writer, h Header, ls *ledger.LedgerSet, init InitialState) error {
	gz := gzip.NewWriter(w)
	defer gz.Close()
	bw := bufio.NewWriter(gz)
	defer bw.Flush()

	events := make([]json.RawMessage, 0, len(ls.CommittedLog()))
	for _, c := range ls.CommittedLog() {
		raw, err := protojson.MarshalOptions{UseProtoNames: true, EmitUnpopulated: false}.Marshal(c)
		if err != nil {
			return fmt.Errorf("manifest: marshal committed event: %w", err)
		}
		events = append(events, raw)
	}
	man := Manifest{
		SchemaVersion:  SchemaVersion,
		KernelVersion:  h.KernelVersion,
		BuildHash:      h.BuildHash,
		InitialState:   init,
		ModelVersions:  h.ModelVersions,
		Events:         events,
		FinalChainHash: hex.EncodeToString(ls.ChainHead()),
	}
	enc := json.NewEncoder(bw)
	enc.SetIndent("", "  ")
	if err := enc.Encode(man); err != nil {
		return fmt.Errorf("manifest: encode: %w", err)
	}
	return nil
}

// Read parses a gzipped manifest from r.
func Read(r io.Reader) (*Manifest, error) {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return nil, fmt.Errorf("manifest: gzip reader: %w", err)
	}
	defer gz.Close()
	dec := json.NewDecoder(gz)
	var man Manifest
	if err := dec.Decode(&man); err != nil {
		return nil, fmt.Errorf("manifest: decode: %w", err)
	}
	if man.SchemaVersion == 0 {
		return nil, fmt.Errorf("manifest: missing schema_version")
	}
	return &man, nil
}

// Events decodes the manifest's events into typed CommittedEvent values.
func (m *Manifest) DecodeEvents() ([]*ledgerv1.CommittedEvent, error) {
	out := make([]*ledgerv1.CommittedEvent, 0, len(m.Events))
	for i, raw := range m.Events {
		c := &ledgerv1.CommittedEvent{}
		opts := protojson.UnmarshalOptions{DiscardUnknown: true}
		if err := opts.Unmarshal(raw, c); err != nil {
			return nil, fmt.Errorf("manifest: decode event %d: %w", i, err)
		}
		out = append(out, c)
	}
	return out, nil
}

// Replay re-applies every event from m against fresh, returning the
// resulting chain head. Returns an error if any event fails to apply,
// or if the resulting chain head does not match the manifest's
// final_chain_hash.
func Replay(m *Manifest, fresh *ledger.LedgerSet) ([]byte, error) {
	events, err := m.DecodeEvents()
	if err != nil {
		return nil, err
	}
	for i, c := range events {
		if err := fresh.AppendReplay(c); err != nil {
			return nil, fmt.Errorf("manifest: replay event %d (global_commit_seq=%d): %w", i, c.GetGlobalCommitSeq(), err)
		}
	}
	got := fresh.ChainHead()
	want, err := hex.DecodeString(m.FinalChainHash)
	if err != nil {
		return nil, fmt.Errorf("manifest: decode final_chain_hash: %w", err)
	}
	if !bytesEqual(got, want) {
		return nil, fmt.Errorf("manifest: replay chain head mismatch:\n  manifest=%x\n  replay  =%x", want, got)
	}
	return got, nil
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
