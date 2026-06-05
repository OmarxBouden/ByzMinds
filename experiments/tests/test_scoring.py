"""Tests for full-sequence A/B log-likelihood scoring (Gate 2 root cause #3).

``experiments._scoring`` scores each candidate continuation by its summed
token log-probability (lm-eval-harness ``loglikelihood`` style). The
decisive property over the earlier first-token comparison: it discriminates
options that share a prefix. ``malformed`` is structurally always False.
All pure tests run without torch; the HF path is torch/transformers-gated.
"""

from __future__ import annotations

import pytest

from experiments._scoring import decide, score_ab_from_logprobs, score_binary_choice_hf


# --- pure decision logic ---------------------------------------------


def test_decide_prefers_higher_logprob():
    r = decide(-1.1, -3.0)
    assert r["chose_a"] is True
    assert r["margin"] == pytest.approx(1.9)
    assert r["malformed"] is False


def test_decide_b_preferred():
    r = decide(-3.0, -0.2)
    assert r["chose_a"] is False
    assert r["margin"] < 0
    assert r["malformed"] is False


def test_decide_tie_resolves_to_a():
    r = decide(-1.0, -1.0)
    assert r["chose_a"] is True and r["margin"] == 0 and r["malformed"] is False


# --- the prefix-sharing fix ------------------------------------------


def test_full_sequence_discriminates_shared_prefix_options():
    """Options share their first two tokens ("I publicly ...") — identical
    first-token logprobs — but the differentiating final token favors A.
    A first-token-only comparison would TIE; full-sequence picks A."""
    shared = [-0.5, -0.5]
    cont_a = shared + [-0.1]   # "... endorse"  (sum -1.1)
    cont_b = shared + [-2.0]   # "... raise"    (sum -3.0)
    r = score_ab_from_logprobs(cont_a, cont_b)
    assert r["chose_a"] is True
    assert r["malformed"] is False
    # And the first tokens alone are uninformative (the bug we fixed):
    assert cont_a[0] == cont_b[0]


def test_full_sequence_picks_b_when_b_more_likely():
    r = score_ab_from_logprobs([-0.5, -2.0], [-0.5, -0.1])
    assert r["chose_a"] is False
    assert r["malformed"] is False


# --- HF path (needs torch + transformers) ----------------------------


def test_score_binary_choice_hf_on_distilgpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    rec = score_binary_choice_hf(
        model, tok, "The capital of France is", " Paris", " banana123 xyzzy"
    )
    # A coherent continuation should outscore gibberish.
    assert rec["chose_a"] is True
    assert rec["malformed"] is False


def test_score_binary_choice_hf_rejects_empty_option():
    with pytest.raises(ValueError, match="empty option"):
        score_binary_choice_hf(None, None, "prompt", "", "B")
