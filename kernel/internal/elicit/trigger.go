// Package elicit implements the scheduler's per-K-tick elicitation
// pass per byzminds-template-spec.md §5.
//
// Trigger flow (driven by scheduler in phase 3.5, between action
// commit and L_cog_ind snapshot):
//
//   1. After the action envelope commits for one agent at tick t, the
//      scheduler asks ShouldElicit(t, K) whether to elicit.
//   2. On yes the scheduler builds an ElicitationRequest payload,
//      commits it to L_ctrl (kernel-signed), then dispatches a
//      second View to the agent's transport with View.elicit_request
//      populated. The agent's transport must respond with a
//      DeclareIntent envelope; non-compliant transports (Step 2 stubs
//      that don't know about elicit) emit some other event_type and
//      the scheduler records a MalformedSubmission rather than
//      committing to L_cog_eli.
//   3. Committed DeclareIntents land on L_cog_eli via the standard
//      LedgerSet.Commit path (schema.Validate routes DeclareIntent →
//      L_cog_eli).
//
// Same-tick ordering: action View FIRST, then elicit View. Replay
// correctness depends on this — the scheduler iterates phase 3.5
// only after phase 3's sorted commit pass finishes.
package elicit

import (
	"fmt"

	"google.golang.org/protobuf/proto"

	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

// DefaultKElicit matches the Step 4 brief judgment call #2.
const DefaultKElicit uint32 = 3

// ShouldElicit returns true iff the scheduler should run an elicit
// pass at tick `t` given the scenario's K_elicit. K==0 disables
// elicitation entirely. K==1 elicits every tick. K==3 (the default)
// elicits at ticks 0, 3, 6, 9, ... in a 15-tick scenario for ~5
// elicitation points per agent.
func ShouldElicit(t uint64, k uint32) bool {
	if k == 0 {
		return false
	}
	return t%uint64(k) == 0
}

// RenderActionSummary produces the short string the elicit prompt
// quotes back to the agent: "{event_type} {key=value, ...}". Same
// renderer regardless of scenario so cross-scenario Δ_cog is
// comparable. Kernel-owned per the brief.
func RenderActionSummary(env *eventsv1.EventEnvelope) (string, error) {
	if env == nil {
		return "", fmt.Errorf("elicit: nil envelope")
	}
	et := env.GetEventType()
	switch et {
	case "Speak":
		msg := &eventsv1.Speak{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s channel=%s content=%q`, et, msg.GetChannelId(), msg.GetContent()), nil
	case "Vote":
		msg := &eventsv1.Vote{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s option=%s`, et, msg.GetOption()), nil
	case "Yield":
		msg := &eventsv1.Yield{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s reason=%q`, et, msg.GetReason()), nil
	case "Yield_Kernel_Synthesized":
		// The agent doesn't know it was timed out; describe in the
		// elicit prompt as a plain Yield so the elicit pass measures
		// the steered reasoning, not the timeout incident.
		return `Yield reason="tick_timeout"`, nil
	case "OpenChannelReq":
		msg := &eventsv1.OpenChannelReq{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s proposed_members=%v`, et, msg.GetProposedMembers()), nil
	case "CloseChannelReq":
		msg := &eventsv1.CloseChannelReq{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s channel_id=%s`, et, msg.GetChannelId()), nil
	case "RequestCapability":
		msg := &eventsv1.RequestCapability{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s cap_id=%s`, et, msg.GetCapId()), nil
	case "DropCapability":
		msg := &eventsv1.DropCapability{}
		if err := proto.Unmarshal(env.GetPayload(), msg); err != nil {
			return et, nil
		}
		return fmt.Sprintf(`%s cap_id=%s`, et, msg.GetCapId()), nil
	default:
		return et, nil
	}
}
