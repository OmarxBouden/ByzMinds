"""LLM call cache for replay determinism.

Original Step 3 key (kept for backward compatibility):

    SHA-256(L0 || L1 || L2 || tool_schemas_json || sampling_params_json
            || theta_serialized || repeng_layer || model_version_hash)

Step 5 milestone 5 brief Decision 4 extends this with:

    || dial_alphas_serialized   # α_d for each dial, dial-keyed
    || layer_assignments_serialized   # which dial → which layer
    || model_revision_sha   # HF revision SHA (was model_version)
    || vllm_version   # e.g. "0.6.4"

When θ changes (different α or different dial composition), the key
changes even if the prompt is identical — same prompt produces
different outputs under different steering, so the cache must
distinguish them. The new fields default to None/empty so existing
Step 3-era cache entries continue to round-trip; Step 5-era callers
populate them.

Storage backend: ``diskcache.Cache`` rooted at the directory the
caller supplies (typically ``runs/<manifest_hash>/llm_cache/``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import diskcache


@dataclass(frozen=True)
class CacheKey:
    """All inputs that flow into the forward-pass cache key."""

    l0: str
    l1: str
    l2: str
    tool_schemas: list[dict] | str  # OpenAI-style tools list or canonical str
    sampling_params: dict
    theta: list[float]
    repeng_layer: int | dict[str, int]  # int for global, dict for per-dial
    model_version: str

    # ---- Step 5 milestone 5 additions (Decision 4) ----
    # Each defaults to None/"" so existing Step 3-era CacheKey
    # constructions still produce the same digest they did before;
    # Step 5-era callers populate the fields so cache invalidation
    # on θ / layer / model / vLLM changes is enforced.
    dial_alphas: dict[str, float] | None = None
    layer_assignments: dict[str, int] | None = None
    model_revision_sha: str = ""
    vllm_version: str = ""

    def digest(self) -> str:
        """SHA-256 hex of the canonical concatenation."""
        h = hashlib.sha256()
        for chunk in (
            self.l0.encode("utf-8"),
            b"\x00",
            self.l1.encode("utf-8"),
            b"\x00",
            self.l2.encode("utf-8"),
            b"\x00",
            json.dumps(self.tool_schemas, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            b"\x00",
            json.dumps(self.sampling_params, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            b"\x00",
            json.dumps(self.theta, separators=(",", ":")).encode("utf-8"),
            b"\x00",
            json.dumps(self.repeng_layer, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            b"\x00",
            self.model_version.encode("utf-8"),
        ):
            h.update(chunk)
        # Step 5 m5 extension — appended only when populated so old
        # tests (and old cache contents) round-trip without change.
        if (
            self.dial_alphas is not None
            or self.layer_assignments is not None
            or self.model_revision_sha
            or self.vllm_version
        ):
            for chunk in (
                b"\x00m5\x00",
                json.dumps(
                    self.dial_alphas or {}, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
                b"\x00",
                json.dumps(
                    self.layer_assignments or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                b"\x00",
                self.model_revision_sha.encode("utf-8"),
                b"\x00",
                self.vllm_version.encode("utf-8"),
            ):
                h.update(chunk)
        return h.hexdigest()


@dataclass
class CachedResponse:
    """One cached forward-pass result.

    ``raw_text`` is the model's stringified output (whatever the
    backend emitted before tool-call extraction). ``tool_call`` is the
    parsed JSON dict if the backend produced a structured call, else
    ``None`` — Route 2 extraction lives in ``tools/validate.py``.
    """

    raw_text: str
    tool_call: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMCache:
    """A diskcache-backed key→CachedResponse store.

    Usage:

        cache = LLMCache(Path("runs/abc/llm_cache"))
        key = CacheKey(...)
        hit = cache.get(key)
        if hit is None:
            hit = backend.generate(...)
            cache.put(key, hit)

    The cache is per-run; sharing a directory across runs collides keys
    only when every input field matches (which is the design intent —
    identical inputs should reuse the same response).
    """

    def __init__(self, root: Path | str, size_limit_bytes: int = 16 * 1024 * 1024 * 1024) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.root), size_limit=size_limit_bytes)

    def get(self, key: CacheKey) -> CachedResponse | None:
        raw = self._cache.get(key.digest())
        if raw is None:
            return None
        return CachedResponse(**raw)

    def put(self, key: CacheKey, response: CachedResponse) -> None:
        self._cache[key.digest()] = {
            "raw_text": response.raw_text,
            "tool_call": response.tool_call,
            "extra": response.extra,
        }

    def __contains__(self, key: CacheKey) -> bool:
        return key.digest() in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._cache.close()

    def get_or_compute(
        self,
        key: CacheKey,
        compute: Callable[[], CachedResponse],
    ) -> CachedResponse:
        """Cache-aside helper. ``compute`` is invoked exactly when the
        key is absent; the returned ``CachedResponse`` is written and
        returned.
        """
        hit = self.get(key)
        if hit is not None:
            return hit
        fresh = compute()
        self.put(key, fresh)
        return fresh
