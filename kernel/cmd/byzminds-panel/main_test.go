package main

import "testing"

// TestDialOrderMatchesPython pins the Go theta ordering to the canonical
// byzminds_agent.DIALS order. If this changes, the Python side must change too
// (and vice versa), or L_cog_ind theta indices will be misattributed.
func TestDialOrderMatchesPython(t *testing.T) {
	want := []string{"authority", "bandwagon", "sycophancy", "free_ride", "collude", "deceive"}
	if len(dialOrder) != len(want) {
		t.Fatalf("dialOrder len %d != %d", len(dialOrder), len(want))
	}
	for i := range want {
		if dialOrder[i] != want[i] {
			t.Fatalf("dialOrder[%d]=%q want %q (MUST mirror Python byzminds_agent.DIALS)", i, dialOrder[i], want[i])
		}
	}
}

func TestThetaForPersona(t *testing.T) {
	th, err := thetaForPersona("collude", "strong")
	if err != nil {
		t.Fatal(err)
	}
	if th[4] != 1.0 { // collude is index 4
		t.Fatalf("collude/strong theta[4]=%v want 1.0", th[4])
	}
	for i, v := range th {
		if i != 4 && v != 0 {
			t.Fatalf("theta[%d]=%v want 0", i, v)
		}
	}
	honest, _ := thetaForPersona("", "none")
	for i, v := range honest {
		if v != 0 {
			t.Fatalf("honest theta[%d]=%v want 0", i, v)
		}
	}
}

func TestParseAgentTheta(t *testing.T) {
	m, err := parseAgentTheta("reviewer_01=collude:strong,reviewer_03=authority:moderate")
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 {
		t.Fatalf("want 2 agents, got %d", len(m))
	}
	if m["reviewer_01"][4] != 1.0 {
		t.Fatalf("reviewer_01 collude/strong theta[4]=%v want 1.0", m["reviewer_01"][4])
	}
	if m["reviewer_03"][0] != 2.0/3.0 { // authority idx 0, moderate
		t.Fatalf("reviewer_03 authority/moderate theta[0]=%v want 0.667", m["reviewer_03"][0])
	}
	if _, err := parseAgentTheta("bad_entry_no_eq"); err == nil {
		t.Fatal("expected error on malformed entry")
	}
}
