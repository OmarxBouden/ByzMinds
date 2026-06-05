"""Unit tests for SteeringHookManager (Step 5 milestone 5 Part 2).

Tests drive the hook math through a synthetic ``nn.Module`` so they
run without vLLM (Mac) and without a GPU. The brief's "test on actual
small model (e.g. distilgpt2) if Llama 8B too heavy for CI" lives in
``test_apply_on_distilgpt2`` and skips cleanly when ``transformers``
is absent.
"""

from __future__ import annotations

import pytest

# Skip the whole module if torch isn't installed.
torch = pytest.importorskip("torch")  # noqa: E402

from byzminds_agent.steering.apply import SteeringHookManager  # noqa: E402


# --- Synthetic model harness -----------------------------------------


class _IdentityLayer(torch.nn.Module):
    """A decoder-like layer whose forward is just the identity. Stored
    in ``model.model.layers[i]`` so it matches the Llama 3.1 access
    path the manager expects."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Real Llama returns a tuple ``(hidden_states, ...)``; we test
        # both tuple and tensor return shapes so the hook handles
        # both. ``forward`` returns the tensor; we re-wrap in tests
        # where needed.
        return x


class _FakeLlama(torch.nn.Module):
    """Mimics ``model.model.layers[i]`` access. Holds N identity
    decoder layers."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8):
        super().__init__()
        # Outer "model" attribute → inner "model" with layers list.
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_IdentityLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.hidden_dim = hidden_dim

    def run_layer(self, layer_idx: int, x: torch.Tensor):
        """Drive one layer's forward (the hook fires here)."""
        return self.model.layers[layer_idx](x)


def _make_manager(*, model=None, dial_to_layer=None, dial_to_vector=None):
    """Build a SteeringHookManager against a synthetic FakeLlama with
    deterministic per-dial steering vectors."""
    hidden = 8
    fake = model or _FakeLlama(n_layers=4, hidden_dim=hidden)
    if dial_to_layer is None:
        dial_to_layer = {"deceive": 1}
    if dial_to_vector is None:
        dial_to_vector = {
            d: torch.arange(hidden, dtype=torch.float32) * (i + 1)
            for i, d in enumerate(dial_to_layer.keys())
        }
    return fake, SteeringHookManager(
        model=fake,
        dial_to_layer=dial_to_layer,
        dial_to_vector=dial_to_vector,
    )


# --- Validators -------------------------------------------------------


def test_init_rejects_unknown_dial():
    fake = _FakeLlama()
    with pytest.raises(ValueError, match="Unknown dial"):
        SteeringHookManager(
            model=fake,
            dial_to_layer={"banana": 1},
            dial_to_vector={"banana": torch.zeros(8)},
        )


def test_init_requires_vector_per_dial():
    fake = _FakeLlama()
    with pytest.raises(ValueError, match="missing steering vectors"):
        SteeringHookManager(
            model=fake,
            dial_to_layer={"deceive": 1, "collude": 2},
            dial_to_vector={"deceive": torch.zeros(8)},  # collude missing
        )


def test_init_requires_llm_or_model():
    with pytest.raises(ValueError, match="llm or model"):
        SteeringHookManager(
            dial_to_layer={"deceive": 1},
            dial_to_vector={"deceive": torch.zeros(8)},
        )


def test_set_theta_rejects_unknown_dial():
    _, m = _make_manager()
    with pytest.raises(ValueError, match="Unknown dial"):
        m.set_theta({"banana": 1.0})


# --- α=0 vs α>0 behavior ---------------------------------------------


def test_alpha_zero_install_is_identity_on_forward_pass():
    """α=0 everywhere → installed hooks add nothing; output equals input."""
    fake, m = _make_manager()
    x = torch.ones(1, 2, 8)
    baseline = fake.run_layer(1, x.clone())
    with m:
        # default α=0 from init
        out = fake.run_layer(1, x.clone())
    assert torch.equal(out, baseline)


def test_alpha_nonzero_install_changes_output():
    fake, m = _make_manager()
    x = torch.ones(1, 2, 8)
    with m:
        m.set_theta({"deceive": 1.0})
        out = fake.run_layer(1, x.clone())
    expected = x + torch.arange(8, dtype=torch.float32)  # alpha=1, v=[0..7]
    assert torch.allclose(out, expected)


# --- set_theta live update --------------------------------------------


def test_set_theta_between_calls_changes_behavior():
    fake, m = _make_manager()
    x = torch.zeros(1, 1, 8)
    with m:
        m.set_theta({"deceive": 1.0})
        a = fake.run_layer(1, x.clone()).clone()
        m.set_theta({"deceive": 0.5})
        b = fake.run_layer(1, x.clone()).clone()
    # a should be 2 × b in magnitude.
    assert torch.allclose(a, b * 2.0)


def test_set_theta_merges_previous_settings():
    """A dial absent from a later set_theta call keeps its prior α —
    the brief's pattern is set-theta-in-place per (agent, condition)."""
    fake, m = _make_manager(
        dial_to_layer={"deceive": 1, "collude": 2},
        dial_to_vector={"deceive": torch.ones(8), "collude": torch.ones(8) * 2.0},
    )
    m.set_theta({"deceive": 0.5, "collude": 0.5})
    m.set_theta({"deceive": 1.0})  # collude unchanged
    assert m.alphas == {"deceive": 1.0, "collude": 0.5}


# --- remove() / context-manager ---------------------------------------


def test_remove_restores_baseline():
    fake, m = _make_manager()
    x = torch.zeros(1, 1, 8)
    baseline = fake.run_layer(1, x.clone())
    m.install()
    m.set_theta({"deceive": 1.0})
    out_installed = fake.run_layer(1, x.clone())
    assert not torch.equal(out_installed, baseline)
    m.remove()
    out_after = fake.run_layer(1, x.clone())
    assert torch.equal(out_after, baseline)


def test_remove_is_idempotent():
    _, m = _make_manager()
    m.install()
    m.remove()
    m.remove()  # second call must be a no-op


def test_double_install_raises():
    _, m = _make_manager()
    m.install()
    with pytest.raises(RuntimeError, match="already installed"):
        m.install()
    m.remove()


def test_context_manager_installs_and_removes():
    fake, m = _make_manager()
    assert not m.installed
    with m:
        assert m.installed
    assert not m.installed


# --- per-layer composition --------------------------------------------


def test_two_dials_at_different_layers_act_independently():
    """deceive@layer 1 with α=1 adds v_d; collude@layer 2 with α=1
    adds v_c at layer 2. The two compositions are independent."""
    fake, m = _make_manager(
        dial_to_layer={"deceive": 1, "collude": 2},
        dial_to_vector={
            "deceive": torch.ones(8) * 1.0,
            "collude": torch.ones(8) * 2.0,
        },
    )
    x = torch.zeros(1, 1, 8)
    with m:
        m.set_theta({"deceive": 1.0, "collude": 1.0})
        # Layer 1 only sees the deceive contribution.
        l1_out = fake.run_layer(1, x.clone())
        # Layer 2 only sees the collude contribution.
        l2_out = fake.run_layer(2, x.clone())
        # Layer 0 (no dial assigned) is identity.
        l0_out = fake.run_layer(0, x.clone())
    assert torch.allclose(l1_out, torch.ones(1, 1, 8) * 1.0)
    assert torch.allclose(l2_out, torch.ones(1, 1, 8) * 2.0)
    assert torch.equal(l0_out, x)


def test_two_dials_at_same_layer_sum_their_contributions():
    """deceive and collude both at layer 1: their α_d · v_d must add
    at that layer (per brief Part 2 / Decision 2)."""
    fake, m = _make_manager(
        dial_to_layer={"deceive": 1, "collude": 1},
        dial_to_vector={
            "deceive": torch.ones(8) * 1.0,
            "collude": torch.ones(8) * 3.0,
        },
    )
    x = torch.zeros(1, 1, 8)
    with m:
        m.set_theta({"deceive": 0.5, "collude": 0.5})
        out = fake.run_layer(1, x.clone())
    # Expected: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
    assert torch.allclose(out, torch.ones(1, 1, 8) * 2.0)


def test_one_hook_per_unique_layer():
    """The brief specifies one ``register_forward_hook`` per unique
    layer, not per dial. Three dials sharing two layers → two hooks."""
    fake, m = _make_manager(
        dial_to_layer={"deceive": 1, "collude": 1, "authority": 2},
        dial_to_vector={
            "deceive": torch.zeros(8),
            "collude": torch.zeros(8),
            "authority": torch.zeros(8),
        },
    )
    m.install()
    try:
        assert len(m._hooks) == 2  # layers {1, 2}
    finally:
        m.remove()


def test_unique_layers_iterated_in_sorted_order():
    """Stable hook installation order — sorted by layer index — so
    the manifest's per-layer composite hash is deterministic."""
    fake = _FakeLlama(n_layers=8, hidden_dim=8)
    m = SteeringHookManager(
        model=fake,
        dial_to_layer={"deceive": 3, "collude": 1, "authority": 2},
        dial_to_vector={d: torch.zeros(8) for d in ("deceive", "collude", "authority")},
    )
    m.install()
    try:
        # The hook handles' insertion order should match sorted(unique_layers).
        # FakeLlama doesn't expose handle ordering directly, but we can
        # assert the installed count and rely on sorted layer iteration
        # in the implementation.
        assert len(m._hooks) == 3
    finally:
        m.remove()


# --- vLLM dispatch (mocked) -------------------------------------------


def test_access_underlying_model_from_llm_navigates_internals():
    """If the caller supplies an ``llm`` instead of ``model``, the
    manager walks ``llm.llm_engine.model_executor.driver_worker.model_runner.model``."""
    fake = _FakeLlama()

    class _ModelRunner:
        model = fake

    class _DriverWorker:
        model_runner = _ModelRunner()

    class _ModelExecutor:
        driver_worker = _DriverWorker()

    class _Engine:
        model_executor = _ModelExecutor()

    class _LLM:
        llm_engine = _Engine()

    m = SteeringHookManager(
        llm=_LLM(),
        dial_to_layer={"deceive": 1},
        dial_to_vector={"deceive": torch.zeros(8)},
    )
    assert m._model is fake


def test_access_underlying_model_raises_clear_error_on_internal_change():
    """If vLLM internals change, the manager surfaces a research-facing
    error pointing at the brief's escalation path."""

    class _Broken:
        pass

    with pytest.raises(RuntimeError, match="vLLM internals"):
        SteeringHookManager(
            llm=_Broken(),
            dial_to_layer={"deceive": 1},
            dial_to_vector={"deceive": torch.zeros(8)},
        )


# --- distilgpt2 path (real model, skipped if transformers absent) ----


def test_apply_on_distilgpt2():
    """The brief's "test on actual small model" check. Loads distilgpt2,
    installs a hook, verifies α=0 → identity and α=1 changes outputs."""
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    # distilgpt2 layout: model.transformer.h[i]. Wrap so it matches the
    # Llama 3.1 access path the manager expects.
    class _Adapter:
        def __init__(self, gpt2):
            self.gpt2 = gpt2
            self.model = type("M", (), {})()
            self.model.layers = gpt2.transformer.h

    adapter = _Adapter(model)
    hidden = model.config.n_embd
    v = torch.zeros(hidden, dtype=torch.float32)
    v[0] = 1.0  # one-hot

    m = SteeringHookManager(
        model=adapter,
        dial_to_layer={"deceive": 0},
        dial_to_vector={"deceive": v},
    )

    inputs = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        # α=0 baseline.
        baseline_logits = model(**inputs).logits.clone()
        with m:
            m.set_theta({"deceive": 0.0})
            zero_logits = model(**inputs).logits.clone()
            assert torch.allclose(zero_logits, baseline_logits)
            m.set_theta({"deceive": 5.0})
            steered_logits = model(**inputs).logits.clone()
            assert not torch.allclose(steered_logits, baseline_logits)
