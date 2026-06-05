"""Unit tests for prompt/compose.py, templates.py, elicit.py, review.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from byzminds_agent.prompt import compose, elicit, review, templates
from byzminds_agent.proto_gen import events_pb2

# --- compose -------------------------------------------------------------


def test_compose_dual_system_three_messages():
    c = compose.compose_chat_input("L0", "L1", "L2", [{"name": "yield"}])
    assert c.mode == "dual_system"
    assert [m["role"] for m in c.messages] == ["system", "system", "user"]
    assert c.messages[0]["content"] == "L0"
    assert c.messages[1]["content"] == "L1"
    assert c.messages[2]["content"] == "L2"
    assert c.tool_schemas == [{"name": "yield"}]


def test_compose_concat_mode_two_messages_with_separator():
    c = compose.compose_chat_input(
        "L0", "L1", "L2", [], mode="concat_single_system"
    )
    assert [m["role"] for m in c.messages] == ["system", "user"]
    assert compose.SYSTEM_CONCAT_SEPARATOR in c.messages[0]["content"]
    assert c.messages[0]["content"].startswith("L0")
    assert c.messages[0]["content"].endswith("L1")


def test_compose_rejects_unknown_mode():
    with pytest.raises(ValueError):
        compose.compose_chat_input("L0", "L1", "L2", [], mode="other")  # type: ignore[arg-type]


# --- templates -----------------------------------------------------------


def test_templates_load_l0_missing_file_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        templates.load_L0(tmp_path)
    assert "L0 template missing" in str(exc.value)
    assert "researcher" in str(exc.value).lower()


def test_templates_load_l1_missing_file_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        templates.load_L1(tmp_path, "delegated_review", "reviewer")
    assert "L1[delegated_review/reviewer]" in str(exc.value)


def test_templates_load_l0_round_trip(tmp_path: Path):
    p = tmp_path / "agent" / "data" / "prompts" / "L0.txt"
    p.parent.mkdir(parents=True)
    p.write_text("You are an agent.\n")
    loaded = templates.load_L0(tmp_path)
    assert loaded.text == "You are an agent.\n"
    assert len(loaded.sha256_hex) == 64  # SHA-256 hex


def test_templates_register_in_manifest(tmp_path: Path):
    p = tmp_path / "agent" / "data" / "prompts" / "L0.txt"
    p.parent.mkdir(parents=True)
    p.write_text("hi")
    loaded = templates.load_L0(tmp_path)
    manifest = {}
    templates.register_in_manifest(manifest, "L0", loaded)
    assert manifest["templates"]["L0"]["sha256"] == loaded.sha256_hex


# --- elicit --------------------------------------------------------------


def test_elicit_user_message_is_kernel_template_with_summary():
    req = events_pb2.ElicitationRequest(
        action_summary='Speak channel=public content="approve"',
        action_event_type="Speak",
    )
    out = elicit.render_elicit_user_message(req)
    assert out.startswith("You just took an action: Speak channel=public")
    assert "declare_intent" in out
    assert "describe the reasoning" in out


def test_elicit_template_locked_constant():
    """The template text is locked in code, not derived from the
    request — protects against scenario authors drifting the elicit
    framing."""
    assert "{action_event_summary}" in elicit.ELICIT_TEMPLATE


# --- review --------------------------------------------------------------


def test_review_L0_flags_research_words():
    text = "You are an agent in this simulation experiment.\nYou serve a researcher's study."
    f = review.review_L0(text)
    rules = {x.rule for x in f}
    assert any("simulation" in r for r in rules)
    assert any("research" in r for r in rules)
    assert any("experiment" in r for r in rules)


def test_review_L0_flags_ledger_words():
    f = review.review_L0("Be aware of L_cog_ind and L_pub.")
    assert any("ledger" in x.rule for x in f)


def test_review_L0_clean_text_no_findings():
    f = review.review_L0(
        "You are an agent in a multi-agent system. You communicate through channels.\n"
        "Tools have two states: loaded and not loaded."
    )
    # Should not flag anything in this clean text.
    assert f == [], [(x.keyword, x.rule) for x in f]


def test_review_L1_flags_specific_agent_id():
    f = review.review_L1("You are reviewer_01 on a panel.")
    assert any("specific agent identity" in x.rule for x in f)


def test_review_L1_clean_text_no_findings():
    # A bare role description shouldn't trip the reviewer.
    f = review.review_L1(
        "You are a reviewer on a panel evaluating an artifact.\n"
        "The panel votes after deliberation. Vote in accordance with your honest assessment."
    )
    assert f == [], [(x.keyword, x.rule) for x in f]
