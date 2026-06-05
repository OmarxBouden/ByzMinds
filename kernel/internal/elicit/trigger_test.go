package elicit

import (
	"testing"

	"google.golang.org/protobuf/proto"

	eventsv1 "github.com/byzminds/byzminds/proto/eventsv1"
)

func TestShouldElicitDefaultK(t *testing.T) {
	cases := []struct {
		tick uint64
		want bool
	}{
		{0, true}, {1, false}, {2, false}, {3, true}, {6, true}, {7, false},
	}
	for _, c := range cases {
		if got := ShouldElicit(c.tick, DefaultKElicit); got != c.want {
			t.Errorf("ShouldElicit(%d, %d) = %v, want %v", c.tick, DefaultKElicit, got, c.want)
		}
	}
}

func TestShouldElicitK0IsDisabled(t *testing.T) {
	for _, tick := range []uint64{0, 1, 5, 100} {
		if ShouldElicit(tick, 0) {
			t.Errorf("ShouldElicit(%d, 0) = true, want false (K=0 disables)", tick)
		}
	}
}

func TestShouldElicitK1IsEveryTick(t *testing.T) {
	for _, tick := range []uint64{0, 1, 2, 3, 100} {
		if !ShouldElicit(tick, 1) {
			t.Errorf("ShouldElicit(%d, 1) = false, want true (K=1 every tick)", tick)
		}
	}
}

func TestRenderActionSummaryForSpeak(t *testing.T) {
	payload, _ := proto.Marshal(&eventsv1.Speak{ChannelId: "public", Content: "hi"})
	env := &eventsv1.EventEnvelope{EventType: "Speak", Payload: payload}
	got, err := RenderActionSummary(env)
	if err != nil {
		t.Fatalf("RenderActionSummary: %v", err)
	}
	want := `Speak channel=public content="hi"`
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRenderActionSummaryForVote(t *testing.T) {
	payload, _ := proto.Marshal(&eventsv1.Vote{Option: "approve"})
	env := &eventsv1.EventEnvelope{EventType: "Vote", Payload: payload}
	got, _ := RenderActionSummary(env)
	if got != "Vote option=approve" {
		t.Errorf("got %q", got)
	}
}

func TestRenderActionSummaryForKernelSynthesizedYield(t *testing.T) {
	env := &eventsv1.EventEnvelope{EventType: "Yield_Kernel_Synthesized"}
	got, _ := RenderActionSummary(env)
	if got != `Yield reason="tick_timeout"` {
		t.Errorf("got %q", got)
	}
}

func TestRenderActionSummaryStable(t *testing.T) {
	payload, _ := proto.Marshal(&eventsv1.Speak{ChannelId: "ch_07", Content: "hi"})
	env := &eventsv1.EventEnvelope{EventType: "Speak", Payload: payload}
	a, _ := RenderActionSummary(env)
	b, _ := RenderActionSummary(env)
	if a != b {
		t.Errorf("not deterministic: %q vs %q", a, b)
	}
}
