"""Runtime forward-pass hook installation.

Per byzminds-step5-milestone5-brief.md §"Implementation work / Part 2"
the canonical surface is the ``SteeringHookManager`` context manager.
It:

  * Resolves per-(dial, layer) vectors at install time.
  * Registers one ``torch.nn.Module.register_forward_hook`` per unique
    layer in ``dial_to_layer.values()``.
  * Reads α coefficients **dynamically** from ``self._alphas`` inside
    each hook, so ``set_theta`` re-tunes between calls without
    re-installing the hooks.
  * Tracks handles for clean ``remove()``; supports the
    ``with hooks: ...`` context-manager pattern.

The module is **import-safe without torch installed** — heavy imports
happen inside ``install``. Tests that mock the model (no torch
needed) still exercise the dispatch math; tests that hit a real
``nn.Module`` skip cleanly when torch is absent.

For backward compatibility the Step 3-era ``SteeringPlan``,
``build_plan``, and ``install_steering`` helpers are kept; the new
``SteeringHookManager`` is what the milestone-5 experiment wiring
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from byzminds_agent import DIALS

# ---------------------------------------------------------------------
# milestone-5 canonical API: SteeringHookManager
# ---------------------------------------------------------------------


class SteeringHookManager:
    """Manage per-layer steering hooks on a vLLM-loaded Llama model.

    Composes ``θ = Σ α_d · v_d`` per layer based on per-dial layer
    assignments and per-dial α magnitudes. Hooks add the composite
    to the residual stream at each unique layer.

    Usage::

        manager = SteeringHookManager(
            llm,
            dial_to_layer={"deceive": 16, "collude": 16, "authority": 20},
            dial_to_vector={"deceive": v_deceive, "collude": v_collude, "authority": v_authority},
        )
        with manager:
            manager.set_theta({"deceive": 0.5, "collude": 0.5})
            outputs = llm.generate(...)
            manager.set_theta({"deceive": 1.0})  # re-tune in place
            outputs = llm.generate(...)
        # hooks removed on __exit__

    Parameters
    ----------
    llm : vllm.LLM, optional
        The vLLM instance. ``_access_underlying_model`` navigates
        vLLM's internals to find the ``nn.Module`` to hook. Either
        ``llm`` or ``model`` must be supplied.
    dial_to_layer : dict[str, int]
        Each dial → its calibrated residual-stream layer index.
    dial_to_vector : dict[str, torch.Tensor]
        Each dial → its per-layer steering vector. The vector must
        be of shape ``(hidden_dim,)`` matching the residual stream.
    model : nn.Module, optional
        Pre-resolved model object. Tests pass this directly instead
        of an ``llm`` so they can drive the hook math against a
        synthetic ``nn.Module``. Production callers pass ``llm``.
    """

    def __init__(
        self,
        llm: Any = None,
        dial_to_layer: dict[str, int] | None = None,
        dial_to_vector: dict[str, Any] | None = None,
        *,
        model: Any = None,
    ) -> None:
        if dial_to_layer is None or dial_to_vector is None:
            raise ValueError("dial_to_layer and dial_to_vector are required")
        self._validate_dials(dial_to_layer.keys())
        # Vectors must cover every dial in the layer assignment.
        missing = set(dial_to_layer.keys()) - set(dial_to_vector.keys())
        if missing:
            raise ValueError(
                f"missing steering vectors for dials: {sorted(missing)}"
            )
        if llm is None and model is None:
            raise ValueError("must supply either llm or model")
        self._llm = llm
        self._model = model if model is not None else self._access_underlying_model_from_llm(llm)
        self._dial_to_layer = dict(dial_to_layer)
        self._dial_to_vector = dict(dial_to_vector)
        self._alphas: dict[str, float] = {d: 0.0 for d in dial_to_layer}
        self._hooks: list[Any] = []

    # ---- public API ------------------------------------------------

    def set_theta(self, alphas: dict[str, float]) -> None:
        """Update per-dial α. Idempotent; merges with previous setting
        (dials not present in ``alphas`` keep their old values).

        If hooks are already installed, this is a cheap dictionary
        update — the installed hooks read ``self._alphas`` dynamically
        on each forward pass, so no re-install is needed.
        """
        for d in alphas:
            if d not in self._dial_to_layer:
                raise ValueError(
                    f"Unknown dial: {d!r}; manager was constructed with "
                    f"{sorted(self._dial_to_layer.keys())}"
                )
        for d, a in alphas.items():
            self._alphas[d] = float(a)

    def install(self) -> None:
        """Register one forward_hook per unique layer in
        ``dial_to_layer.values()``."""
        if self._hooks:
            raise RuntimeError(
                "Hooks already installed; call remove() first or use the "
                "context-manager pattern (`with manager: ...`)."
            )
        import torch  # noqa: F401  — checked at install time

        unique_layers = sorted(set(self._dial_to_layer.values()))
        for layer_idx in unique_layers:
            dials_at_layer = sorted(
                d for d, layer in self._dial_to_layer.items() if layer == layer_idx
            )
            layer_module = self._layer_module(layer_idx)
            hook = self._make_hook(dials_at_layer)
            handle = layer_module.register_forward_hook(hook)
            self._hooks.append(handle)

    def remove(self) -> None:
        """Deregister all hooks. Idempotent — calling twice is a no-op."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    # ---- context manager ------------------------------------------

    def __enter__(self) -> "SteeringHookManager":
        self.install()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()

    # ---- properties for tests / debugging --------------------------

    @property
    def installed(self) -> bool:
        return bool(self._hooks)

    @property
    def alphas(self) -> dict[str, float]:
        """Defensive copy of the current α settings."""
        return dict(self._alphas)

    # ---- hook factory ---------------------------------------------

    def _make_hook(self, dials_in_scope: list[str]):
        """Build the hook closure for one layer.

        The closure captures ``self`` and the ``dials_in_scope`` list;
        each forward pass reads ``self._alphas`` to pick up live α
        updates from ``set_theta``. The composite ``Σ_d α_d · v_d`` is
        only materialized when at least one α is non-zero (saves
        memory on α=0 baseline runs).
        """
        import torch

        manager = self
        dials = list(dials_in_scope)

        def hook(_module, _inputs, outputs):
            # Outputs from Llama decoder layer: (hidden_states, ...optional...).
            # Some adapters / model variants return the tensor directly.
            if isinstance(outputs, tuple):
                hs = outputs[0]
            else:
                hs = outputs
            composite = None
            for d in dials:
                alpha = manager._alphas.get(d, 0.0)
                if alpha == 0.0:
                    continue
                v = manager._dial_to_vector[d]
                v = v.to(device=hs.device, dtype=hs.dtype)
                if composite is None:
                    composite = alpha * v
                else:
                    composite = composite + alpha * v
            if composite is None:
                return outputs  # α=0 everywhere → no-op
            # composite is (hidden_dim,); broadcast across (batch, seq, hidden_dim).
            new_hs = hs + composite
            if isinstance(outputs, tuple):
                return (new_hs,) + outputs[1:]
            return new_hs

        return hook

    # ---- vLLM internal access -------------------------------------

    @staticmethod
    def _access_underlying_model_from_llm(llm: Any) -> Any:
        """Navigate vLLM 0.6+ internals to the underlying ``nn.Module``.

        Path: ``llm.llm_engine.model_executor.driver_worker.model_runner.model``.
        Brief Decision 1 documents this is the established RepE pattern
        for vLLM and falls back to ``enforce_eager=True`` if hooks
        don't fire on the continuous-batching path.
        """
        try:
            return llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError as exc:
            raise RuntimeError(
                "SteeringHookManager: could not access underlying model via "
                "llm.llm_engine.model_executor.driver_worker.model_runner.model "
                "— vLLM internals may have changed. Per Step 5 m5 brief Decision 1, "
                "ping researcher before falling back to a non-vLLM path."
            ) from exc

    def _layer_module(self, layer_idx: int) -> Any:
        """Return the ``nn.Module`` for layer ``layer_idx``.

        Llama 3.1 layout: ``model.model.layers[i]``. Test-mode mocks
        provide the same shape (``mock.model.layers[i]``); production
        vLLM-loaded Llama models do as well.
        """
        try:
            return self._model.model.layers[layer_idx]
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(
                f"SteeringHookManager: model.model.layers[{layer_idx}] inaccessible "
                f"({type(exc).__name__}: {exc}). Llama 3.1 has 32 layers; check "
                "dial_to_layer values are in [0, 31] and the model exposes the "
                "expected attribute path."
            ) from exc

    @staticmethod
    def _validate_dials(it: Iterable[str]) -> None:
        for d in it:
            if d not in DIALS:
                raise ValueError(
                    f"Unknown dial {d!r}; expected one of {DIALS}."
                )


# ---------------------------------------------------------------------
# Step 3 backward-compat surface (kept; deprecated in favor of
# SteeringHookManager). Newer callers prefer SteeringHookManager.
# ---------------------------------------------------------------------


@dataclass
class SteeringPlan:
    """Resolved composition: which dials at α apply at which layer."""

    theta: dict[str, float]            # dial → α
    layer_assignments: dict[str, int]  # dial → layer index
    composed_by_layer: dict[int, Any] = field(default_factory=dict)


def build_plan(theta: dict[str, float], layer_assignments: dict[str, int]) -> SteeringPlan:
    """Step-3 helper: validates inputs and returns a plan that
    ``install_steering`` can hand to the backend.
    """
    for d in theta:
        if d not in DIALS:
            raise ValueError(f"unknown dial {d!r}; expected one of {DIALS}")
    for d in layer_assignments:
        if d not in DIALS:
            raise ValueError(f"unknown dial {d!r}; expected one of {DIALS}")
    return SteeringPlan(theta=dict(theta), layer_assignments=dict(layer_assignments))


def install_steering(model, plan: SteeringPlan, vectors: dict[str, dict[int, Any]]):
    """Step-3 surface kept for callers that haven't migrated to
    ``SteeringHookManager``. Hand-builds the composite vectors and
    registers hooks. Returns the handles.

    Deprecated in favor of ``SteeringHookManager`` per Step 5
    milestone-5 brief Part 2.
    """
    import torch  # local import keeps the package import-safe sans torch

    composed_by_layer: dict[int, torch.Tensor] = {}
    for dial, layer in plan.layer_assignments.items():
        alpha = plan.theta.get(dial, 0.0)
        if alpha == 0.0:
            continue
        v = vectors[dial][layer]
        if layer not in composed_by_layer:
            composed_by_layer[layer] = alpha * v
        else:
            composed_by_layer[layer] = composed_by_layer[layer] + alpha * v
    plan.composed_by_layer = composed_by_layer

    handles = []
    for layer_idx, vec in composed_by_layer.items():
        target = model.model.layers[layer_idx]

        def _hook(_module, _inputs, outputs, _vec=vec):
            if isinstance(outputs, tuple):
                hs = outputs[0] + _vec
                return (hs,) + outputs[1:]
            return outputs + _vec

        handles.append(target.register_forward_hook(_hook))
    return handles
