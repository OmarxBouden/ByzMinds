"""A/B **full-sequence** log-likelihood scoring for binary forced-choice probes.

Mirrors lm-evaluation-harness's ``loglikelihood`` approach: score each
candidate continuation by the model's summed token log-probability under
teacher forcing, and pick the higher one. This replaces (a) the brittle
substring matching that conflated malformed output with 50/50 behavior,
and (b) the earlier *first-token-only* A/B comparison, which cannot
discriminate options that share a prefix — e.g. deceive's
"I publicly endorse …" vs "I publicly raise …" share the first tokens, so
a first-token comparison is blind to them. Full-sequence scoring fixes
this (milestone-5 Gate 2 root cause #3; see byzminds-m5-gate2-diagnosis.md).

Design:
  * ``decide`` / ``score_ab_from_logprobs`` are pure and dependency-free
    (unit-tested without torch).
  * ``seq_logprob_hf`` / ``score_binary_choice_hf`` run on an in-process
    HuggingFace ``transformers`` model (torch imported lazily so the
    module stays import-safe on the scaffold-tier environment). This is
    the production path now that vLLM is out of the steered/agent loop.

Key invariant: ``malformed`` is **structurally always False** — every
prompt yields a definite A/B preference.
"""

from __future__ import annotations

from typing import Sequence

NEG_INF = float("-inf")


def decide(logp_a: float, logp_b: float) -> dict:
    """Pure decision from two (summed) continuation log-probs.

    ``chose_a`` is True iff option A is at least as likely as B (ties
    resolve to A deterministically for reproducibility).
    """
    margin = logp_a - logp_b
    return {
        "chose_a": margin >= 0,
        "logp_a": logp_a,
        "logp_b": logp_b,
        "margin": margin,
        "malformed": False,  # structural invariant
    }


def score_ab_from_logprobs(
    cont_a_token_logprobs: Sequence[float],
    cont_b_token_logprobs: Sequence[float],
) -> dict:
    """Decide between two continuations given their per-token log-probs.

    Sums each continuation's token log-probs (full-sequence likelihood)
    and compares. Backend-agnostic: callers supply the per-token logprobs
    however they were obtained (HF, an API that exposes logprobs, etc.).
    """
    return decide(float(sum(cont_a_token_logprobs)), float(sum(cont_b_token_logprobs)))


def seq_logprob_hf(model, tokenizer, prompt: str, continuation: str) -> float:
    """Summed log-prob of ``continuation`` tokens given ``prompt`` under
    teacher forcing, on an in-process HF causal LM.

    ``prompt`` is expected to already carry the chat template's BOS (e.g.
    rendered via ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``),
    so neither piece adds special tokens.
    """
    import torch  # lazy

    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    c_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([p_ids, c_ids], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(full).logits  # (1, T, V)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    lp = p_ids.shape[1]
    total = 0.0
    for i in range(c_ids.shape[1]):
        tok = int(c_ids[0, i])
        total += float(logprobs[0, lp + i - 1, tok])
    return total


def score_binary_choice_hf(model, tokenizer, prompt: str, option_a: str, option_b: str) -> dict:
    """Full-sequence A/B log-likelihood scoring on an HF model.

    Returns the ``decide(...)`` record. Robust to prefix-sharing options
    and to formatting drift (we never parse generated text).
    """
    if not option_a or not option_b:
        raise ValueError(f"empty option: a={option_a!r} b={option_b!r}")
    lp_a = seq_logprob_hf(model, tokenizer, prompt, option_a)
    lp_b = seq_logprob_hf(model, tokenizer, prompt, option_b)
    return decide(lp_a, lp_b)
