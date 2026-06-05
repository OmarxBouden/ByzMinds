"""Stage B -- HF-transformers activation steering (apply side).

Replaces the vLLM hook path (the Gate-2 fragility/incompat source) with plain
HF-transformers forward hooks, the standard substrate for Persona Vectors / CAA
(personavectors2025, rimsky2024caa). A hook on decoder layer L adds a composite
of per-dial steering vectors to the residual stream:

    h_L  <-  h_L  +  sum_d  alpha_d * v_d^(L)

The vectors are the NATURAL diff-of-means directions from
``byzminds_agent.steering.extract`` (NOT unit-normalized -- the Gate-2 root-cause
fix), so alpha in [0,1] is a meaningful fraction of the true activation gap.
Alphas are read dynamically so a calibration sweep re-tunes without reinstalling.

Used as a context manager:

    with HFSteeringHooks(model, {16: [v_collude_16]}, alpha=1.0):
        out = model.generate(...)         # steered
"""

from __future__ import annotations


def _decoder_layers(model):
    """Return the list of decoder layer modules for a HF causal-LM (Llama/Apertus
    expose ``model.model.layers``; fall back to a base-model attr)."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError("could not find decoder layers (expected model.model.layers)")
    return layers


class HFSteeringHooks:
    """Install/remove residual-stream steering hooks on a HF causal-LM.

    Parameters
    ----------
    model : a HF ``AutoModelForCausalLM``.
    layer_to_vectors : {layer_idx: [tensor(hidden_dim), ...]} -- vectors added at
        that layer (one per active dial). Tensors are the natural-norm directions.
    alpha : the steering coefficient applied to every vector (read live; mutate
        ``self.alpha`` between calls to re-tune).
    """

    def __init__(self, model, layer_to_vectors: dict, alpha: float = 1.0):
        self.model = model
        self.layer_to_vectors = {int(l): list(vs) for l, vs in layer_to_vectors.items()}
        self.alpha = float(alpha)
        self._handles: list = []

    def _make_hook(self, vectors):
        import torch

        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            steer = torch.zeros_like(hs[0, 0])
            for v in vectors:
                steer = steer + self.alpha * v.to(dtype=hs.dtype, device=hs.device)
            hs = hs + steer  # broadcast over (batch, seq)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def install(self):
        if self._handles:
            raise RuntimeError("hooks already installed; remove() first")
        layers = _decoder_layers(self.model)
        for layer_idx, vectors in self.layer_to_vectors.items():
            self._handles.append(layers[layer_idx].register_forward_hook(self._make_hook(vectors)))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False


def _selftest():
    """Tiny CPU self-test of the hook math: steering must change the logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "sshleifer/tiny-gpt2"  # ~few-MB model; tiny-gpt2 has model.h not model.layers
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    # tiny-gpt2 exposes transformer.h; adapt the layer-finder for the self-test
    layers = model.transformer.h
    hdim = model.config.n_embd
    L = len(layers) - 1  # tiny-gpt2 has only 2 layers
    ids = tok("the panel", return_tensors="pt")
    with torch.no_grad():
        base = model(**ids).logits[0, -1]
    v = torch.randn(hdim) * base.new_tensor(5.0).abs()
    handle = layers[L].register_forward_hook(
        lambda m, i, o: (o[0] + 3.0 * v.to(o[0].dtype),) + tuple(o[1:]) if isinstance(o, tuple) else o + 3.0 * v)
    with torch.no_grad():
        steered = model(**ids).logits[0, -1]
    handle.remove()
    delta = (steered - base).abs().max().item()
    print(f"hook self-test: max|Δlogit|={delta:.3f} -> {'OK' if delta > 1e-3 else 'NO EFFECT'}")
    return delta > 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
