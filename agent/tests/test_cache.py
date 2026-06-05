"""Unit tests for cache.py — content-addressable key + diskcache I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

from byzminds_agent.cache import CachedResponse, CacheKey, LLMCache


def _key(**overrides) -> CacheKey:
    defaults = dict(
        l0="L0 placeholder",
        l1="L1 placeholder",
        l2="L2 placeholder",
        tool_schemas=[{"name": "yield"}],
        sampling_params={"temperature": 0.0, "max_tokens": 64},
        theta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        repeng_layer=16,
        model_version="llama-3.1-8b-instruct@bf16#abc",
    )
    defaults.update(overrides)
    return CacheKey(**defaults)


def test_key_digest_stable_across_calls():
    k = _key()
    assert k.digest() == k.digest()
    assert len(k.digest()) == 64  # SHA-256 hex


def test_key_changes_with_every_input():
    base = _key().digest()
    assert _key(l0="different").digest() != base
    assert _key(l1="different").digest() != base
    assert _key(l2="different").digest() != base
    assert _key(tool_schemas=[{"name": "speak"}]).digest() != base
    assert _key(sampling_params={"temperature": 0.7}).digest() != base
    assert _key(theta=[0.5, 0, 0, 0, 0, 0]).digest() != base
    assert _key(repeng_layer=12).digest() != base
    assert _key(repeng_layer={"sycophancy": 16}).digest() != base
    assert _key(model_version="other").digest() != base


def test_diskcache_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = LLMCache(tmp)
        key = _key()
        assert key not in cache
        cache.put(
            key,
            CachedResponse(
                raw_text='{"name":"yield","arguments":{"reason":"x"}}',
                tool_call={"name": "yield", "arguments": {"reason": "x"}},
            ),
        )
        assert key in cache
        hit = cache.get(key)
        assert hit is not None
        assert hit.tool_call == {"name": "yield", "arguments": {"reason": "x"}}
        cache.close()


def test_diskcache_persists_across_handles():
    with tempfile.TemporaryDirectory() as tmp:
        key = _key()
        c1 = LLMCache(tmp)
        c1.put(key, CachedResponse(raw_text="hello"))
        c1.close()
        c2 = LLMCache(tmp)
        hit = c2.get(key)
        assert hit is not None
        assert hit.raw_text == "hello"
        c2.close()


def test_get_or_compute_calls_compute_once():
    with tempfile.TemporaryDirectory() as tmp:
        cache = LLMCache(tmp)
        key = _key()
        calls = {"n": 0}

        def compute() -> CachedResponse:
            calls["n"] += 1
            return CachedResponse(raw_text=f"computed call={calls['n']}")

        a = cache.get_or_compute(key, compute)
        b = cache.get_or_compute(key, compute)
        assert a.raw_text == b.raw_text == "computed call=1"
        assert calls["n"] == 1
        cache.close()


# --- Step 5 milestone 5 extensions (Decision 4) ----------------------


def test_m5_dial_alphas_change_produces_different_key():
    """The brief's central correctness claim: same prompt, different θ
    → different cache key. Without this, the cache silently reuses an
    α=0 response for an α=1 forward pass."""
    base = _key(dial_alphas={"deceive": 0.0, "collude": 0.0})
    shifted = _key(dial_alphas={"deceive": 1.0, "collude": 0.0})
    assert base.digest() != shifted.digest()


def test_m5_layer_assignments_change_produces_different_key():
    """If the same θ is applied at a different layer, the forward pass
    differs and the cache must distinguish them."""
    base = _key(
        dial_alphas={"deceive": 0.5},
        layer_assignments={"deceive": 16},
    )
    different_layer = _key(
        dial_alphas={"deceive": 0.5},
        layer_assignments={"deceive": 20},
    )
    assert base.digest() != different_layer.digest()


def test_m5_model_revision_sha_change_produces_different_key():
    """A HuggingFace re-tag (same model name, different weights SHA)
    must invalidate cache entries — RepE outputs change with the
    underlying weights."""
    base = _key(model_revision_sha="abc123")
    re_tagged = _key(model_revision_sha="def456")
    assert base.digest() != re_tagged.digest()


def test_m5_vllm_version_change_produces_different_key():
    """vLLM 0.6.4 vs 0.6.5 may have different sampling internals;
    cache invalidates so we don't silently mix outputs across builds."""
    base = _key(vllm_version="0.6.4")
    upgraded = _key(vllm_version="0.6.5")
    assert base.digest() != upgraded.digest()


def test_m5_backward_compat_keys_without_new_fields_unchanged():
    """A Step 3-era CacheKey with no m5 fields populated produces the
    same digest before and after the m5 extension lands — existing
    cache entries round-trip cleanly."""
    legacy = _key()  # no m5 fields populated → all defaults
    # The default constructor leaves dial_alphas=None, layer_assignments=None,
    # model_revision_sha="", vllm_version=""; the digest's m5 block is
    # conditionally appended only when at least one is populated. Two
    # legacy keys must agree.
    assert legacy.digest() == _key().digest()


def test_m5_dial_alphas_canonicalized_by_dial_name():
    """``{"deceive": 0.5, "collude": 0.5}`` and the same dict in a
    different insertion order must produce the same digest — the
    cache is invariant to dial-dict iteration order."""
    a = _key(dial_alphas={"deceive": 0.5, "collude": 0.5})
    b = _key(dial_alphas={"collude": 0.5, "deceive": 0.5})
    assert a.digest() == b.digest()


def test_m5_full_key_round_trip_through_diskcache():
    """End-to-end: build a fully-populated m5 key, put a response,
    get it back."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = LLMCache(tmp)
        key = _key(
            dial_alphas={"deceive": 1.0, "collude": 0.5},
            layer_assignments={"deceive": 16, "collude": 16},
            model_revision_sha="0e9e39f249a16976918f6564b8830bc894c89659",
            vllm_version="0.6.4",
        )
        cache.put(key, CachedResponse(raw_text="hello m5"))
        hit = cache.get(key)
        assert hit is not None
        assert hit.raw_text == "hello m5"
        cache.close()
