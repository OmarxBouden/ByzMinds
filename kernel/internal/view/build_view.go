// Package view implements BuildView — the deterministic kernel
// function that projects ledger + handler state into a per-(agent, tick)
// structured View. Per byzminds-template-spec.md §4, it returns the
// TemplateContext analogue, NOT a rendered L2 string (text rendering is
// Step 4's job).
//
// Determinism contract: identical inputs (ledger state at tick t,
// handler state, agent_id) → byte-identical canonical marshaling of
// the returned *View. All collections are sorted (channel_id alpha,
// available_tools alpha, capabilities alpha) and the channel-history
// projection reads in (tick, global_commit_seq) order.
package view

import (
	"errors"
	"fmt"
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
	ledgerv1 "github.com/byzminds/byzminds/proto/ledgerv1"
	viewv1 "github.com/byzminds/byzminds/proto/viewv1"
)

// PublicChannelID is the conventional id of the broadcast channel. It
// is always part of every agent's channel_memberships.
const PublicChannelID = "public"

// BuildView returns the structured per-tick view for one agent.
//
// The caller (typically the scheduler) holds whatever locks it needs;
// BuildView itself reads via the handler's accessor methods and the
// ledger's snapshot APIs, all of which take their own locks briefly.
func BuildView(h *handler.Handler, ls *ledger.LedgerSet, agentID string, tick uint64) (*viewv1.View, error) {
	if h == nil || ls == nil {
		return nil, errors.New("view: nil handler or ledger")
	}
	if _, ok := h.LookupAgent(agentID); !ok {
		return nil, fmt.Errorf("view: unknown agent %q", agentID)
	}

	scn := h.Scenario()
	if scn == nil {
		return nil, errors.New("view: scenario not loaded")
	}

	// 1. agent_self_view
	memberships := append([]string{PublicChannelID}, h.AgentChannelMemberships(agentID)...)
	sort.Strings(memberships)
	caps := h.AgentLoadedCapabilities(agentID)

	// 2. scenario state
	phaseName, round, totalRounds, availableTools := h.PhaseAt(tick)

	// 3. channel histories
	histories := make([]*viewv1.ChannelHistory, 0, len(memberships))
	for _, cid := range memberships {
		kc := scenarioWindow(scn, cid)
		msgs := readChannelHistory(ls, cid, tick, kc)
		histories = append(histories, &viewv1.ChannelHistory{
			ChannelId: cid,
			Messages:  msgs,
		})
	}
	// histories is already alpha-sorted because memberships is sorted.

	// 4. externals
	externals := h.DrainExternalInjects(agentID)
	extMsgs := make([]*viewv1.ExternalMsg, 0, len(externals))
	for _, e := range externals {
		extMsgs = append(extMsgs, &viewv1.ExternalMsg{
			ClaimedSource: e.ClaimedSource,
			Content:       e.Content,
			InjectTick:    e.InjectTick,
		})
	}

	return &viewv1.View{
		AgentId:            agentID,
		Tick:               tick,
		ChannelMemberships: memberships,
		LoadedCapabilities: caps,
		Scenario: &viewv1.ScenarioRef{
			ScenarioName:     scn.Name,
			ScenarioYamlHash: scn.YAMLHash,
		},
		TaskArtifact:     scn.TaskArtifact,
		Phase:            phaseName,
		Round:            round,
		TotalRounds:      totalRounds,
		ChannelHistories: histories,
		ExternalMessages: extMsgs,
		AvailableTools:   availableTools,
	}, nil
}

// CanonicalBytes returns the deterministic protobuf marshaling of v.
// Used by the determinism tests and Experiment 005.
func CanonicalBytes(v *viewv1.View) ([]byte, error) {
	return proto.MarshalOptions{Deterministic: true}.Marshal(v)
}

// readChannelHistory returns the last K Speak messages on channelID with
// envelope.tick <= tick, oldest first.
func readChannelHistory(ls *ledger.LedgerSet, channelID string, tick uint64, K uint32) []*viewv1.Message {
	var l *ledger.Ledger
	if channelID == PublicChannelID {
		l = ls.Pub()
	} else {
		l = ls.Prv(channelID)
	}
	if l == nil {
		return nil
	}
	all := l.Snapshot()
	// Filter to Speak events at tick <= cutoff.
	filtered := make([]*ledgerv1.CommittedEvent, 0, len(all))
	for _, c := range all {
		if c.GetEnvelope().GetTick() > tick {
			continue
		}
		if c.GetEnvelope().GetEventType() != "Speak" {
			continue
		}
		filtered = append(filtered, c)
	}
	// Sort by (tick, global_commit_seq) oldest first (Snapshot is already
	// in commit order; this sort is a safety net for canonical ordering).
	sort.SliceStable(filtered, func(i, j int) bool {
		if filtered[i].GetEnvelope().GetTick() != filtered[j].GetEnvelope().GetTick() {
			return filtered[i].GetEnvelope().GetTick() < filtered[j].GetEnvelope().GetTick()
		}
		return filtered[i].GetGlobalCommitSeq() < filtered[j].GetGlobalCommitSeq()
	})
	// Take last K.
	if K > 0 && uint32(len(filtered)) > K {
		filtered = filtered[uint32(len(filtered))-K:]
	}
	// Project to viewv1.Message.
	out := make([]*viewv1.Message, 0, len(filtered))
	for _, c := range filtered {
		out = append(out, projectSpeak(c))
	}
	return out
}

// projectSpeak builds a viewv1.Message from a committed Speak event.
// sender_id is the agent_id corresponding to envelope.emitter_pubkey.
func projectSpeak(c *ledgerv1.CommittedEvent) *viewv1.Message {
	msg := &eventsv1.Speak{}
	_ = proto.Unmarshal(c.GetEnvelope().GetPayload(), msg)
	return &viewv1.Message{
		// sender_id: filled in by the caller, since the view package
		// does not have access to the handler's pubkey→agent_id map
		// here. We fill it through the AgentMapper wrapper below.
		Content:         msg.GetContent(),
		Tick:            c.GetEnvelope().GetTick(),
		GlobalCommitSeq: c.GetGlobalCommitSeq(),
	}
}

func scenarioWindow(scn *handler.ScenarioState, channelID string) uint32 {
	if k, ok := scn.HistoryWindow[channelID]; ok {
		return k
	}
	if channelID == PublicChannelID {
		if k, ok := scn.HistoryWindow["public"]; ok {
			return k
		}
		return handler.DefaultPublicHistoryWindow
	}
	if k, ok := scn.HistoryWindow["private"]; ok {
		return k
	}
	return handler.DefaultPrivateHistoryWindow
}

// ResolveSenderIDs walks v's channel histories and fills sender_id on
// each Message by mapping the original committed event's
// envelope.emitter_pubkey to an agent_id via the handler. The view
// builder cannot do this inline because Message lacks the pubkey field
// (it is intentionally stripped per the template spec); the caller
// supplies a per-tick scan of L_pub + L_prv channels to resolve.
//
// This is invoked by the scheduler after BuildView returns, so the
// sender_id resolution stays on the kernel side.
func ResolveSenderIDs(v *viewv1.View, h *handler.Handler, ls *ledger.LedgerSet) {
	registered := h.AllRegisteredAgents()
	pkToID := make(map[string]string, len(registered))
	for id, a := range registered {
		pkToID[string(a.Pubkey)] = id
	}
	resolveOnLedger := func(channelID string) map[uint64]string {
		out := map[uint64]string{}
		var l *ledger.Ledger
		if channelID == PublicChannelID {
			l = ls.Pub()
		} else {
			l = ls.Prv(channelID)
		}
		if l == nil {
			return out
		}
		for _, c := range l.Snapshot() {
			if c.GetEnvelope().GetEventType() != "Speak" {
				continue
			}
			id, ok := pkToID[string(c.GetEnvelope().GetEmitterPubkey())]
			if !ok {
				id = "<unknown>"
			}
			out[c.GetGlobalCommitSeq()] = id
		}
		return out
	}
	for _, ch := range v.GetChannelHistories() {
		lookup := resolveOnLedger(ch.GetChannelId())
		for _, m := range ch.GetMessages() {
			if id, ok := lookup[m.GetGlobalCommitSeq()]; ok {
				m.SenderId = id
			}
		}
	}
}
