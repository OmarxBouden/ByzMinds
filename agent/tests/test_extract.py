"""Unit tests for steering vector extraction (post Gate 2 refactor).

Covers the three things the refactor changed in
``byzminds_agent.steering.extract``:

  * vectors are kept at NATURAL norm (no unit normalization) — Gate 2
    root cause #1;
  * contrast-pair sides are rendered through the tokenizer's native
    chat template — Gate 2 root cause #2;
  * the saved payload carries an audit ``_metadata`` block and stays
    backward compatible with ``steering.vectors.load_dial``.

The torch-free tests (parse / render / template-detection) run
everywhere; the tensor-payload and distilgpt2 end-to-end tests skip
cleanly when torch / transformers are absent (scaffold-tier env).
"""

from __future__ import annotations

import json

import pytest

from byzminds_agent.steering import extract

# --- torch-free: contrast-pair parsing -------------------------------


def test_parse_pairs_accepts_plus_minus_and_p_plus_p_minus(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"plus": "A+", "minus": "A-"})
        + "\n"
        + json.dumps({"p_plus": "B+", "p_minus": "B-"})
        + "\n"
    )
    pairs = extract.parse_pairs(p)
    assert pairs == [("A+", "A-"), ("B+", "B-")]


def test_parse_pairs_rejects_missing_field(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"plus": "only-plus"}) + "\n")
    with pytest.raises(ValueError, match="missing 'plus'/'minus'"):
        extract.parse_pairs(p)


def test_parse_pairs_rejects_empty_file(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("# just a comment\n\n")
    with pytest.raises(ValueError, match="empty contrast pair file"):
        extract.parse_pairs(p)


# --- torch-free: chat-template rendering -----------------------------


class _FakeTokenizer:
    """Records apply_chat_template calls; emits Llama-style markers so
    rendering is observable without loading a real model."""

    def __init__(self, *, has_template=True):
        self.chat_template = "FAKE-TEMPLATE" if has_template else None
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        assert tokenize is False
        return "<|begin_of_text|>" + messages[-1]["content"] + "<|eot_id|>ASSISTANT"


def test_chat_template_available_detects_template():
    assert extract.chat_template_available(_FakeTokenizer(has_template=True)) is True
    assert extract.chat_template_available(_FakeTokenizer(has_template=False)) is False


def test_render_for_extraction_routes_through_template():
    tok = _FakeTokenizer(has_template=True)
    out = extract.render_for_extraction(tok, "be honest")
    assert out.startswith("<|begin_of_text|>")
    assert "be honest" in out
    # Routed as a single user turn with add_generation_prompt=True.
    assert tok.calls[0]["messages"] == [{"role": "user", "content": "be honest"}]
    assert tok.calls[0]["add_generation_prompt"] is True


def test_render_for_extraction_falls_back_without_template():
    tok = _FakeTokenizer(has_template=False)
    assert extract.render_for_extraction(tok, "raw text") == "raw text"
    assert tok.calls == []  # never called the template path


# --- payload + metadata (needs torch) --------------------------------


def test_build_save_payload_keeps_natural_norm_and_metadata():
    torch = pytest.importorskip("torch")
    vectors = {
        8: torch.arange(4, dtype=torch.float32) * 3.0,  # norm well above 1.0
        16: torch.ones(4, dtype=torch.float32) * 5.0,
    }
    payload = extract.build_save_payload(
        vectors, model_id="m", last_k_tokens=4, chat_template_applied=True
    )
    # layer tensors preserved verbatim (NOT renormalized)
    assert torch.equal(payload["layer_8"], vectors[8])
    meta = payload["_metadata"]
    assert meta["unit_normalized"] is False
    assert meta["chat_template_applied"] is True
    assert meta["model_id"] == "m"
    # natural norms recorded and clearly > 1.0 (the whole point of the fix)
    assert meta["natural_norms"]["16"] == pytest.approx(10.0)  # ‖[5,5,5,5]‖ = 10
    assert all(n > 1.0 for n in meta["natural_norms"].values())


def test_payload_roundtrips_and_load_dial_ignores_metadata(tmp_path):
    torch = pytest.importorskip("torch")

    vectors = {8: torch.ones(4) * 2.0, 12: torch.ones(4) * 4.0}
    payload = extract.build_save_payload(
        vectors, model_id="m", last_k_tokens=4, chat_template_applied=True
    )
    f = tmp_path / "deceive.pt"
    torch.save(payload, f)

    raw = torch.load(f, map_location="cpu", weights_only=True)
    # load_dial-style filtering: only layer_* keys become layer ints.
    loaded = {
        int(k.removeprefix("layer_")): v for k, v in raw.items() if k.startswith("layer_")
    }
    assert set(loaded) == {8, 12}
    assert "_metadata" in raw and not any(k == "_metadata" for k in loaded)


# --- end-to-end natural-norm extraction (needs transformers) ---------


def test_extract_one_dial_natural_norm_on_distilgpt2(tmp_path, monkeypatch):
    """The brief's 'extract on a small model' check: vectors must have
    natural norm (not ~1.0). Skips when transformers is absent."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    # Point the extractor at a tiny temp contrast-pair file.
    pairs_dir = tmp_path / "contrast_pairs"
    pairs_dir.mkdir()
    (pairs_dir / "deceive.jsonl").write_text(
        "\n".join(
            json.dumps({"plus": f"deceive {i}", "minus": f"honest {i}"}) for i in range(5)
        )
    )
    monkeypatch.setattr(extract, "CONTRAST_PAIRS_DIR", pairs_dir)

    vectors = extract.extract_one_dial(
        "deceive", [2, 4], model=model, tokenizer=tok, last_k_tokens=4
    )
    # The decisive assertion: not unit-normalized.
    for layer, v in vectors.items():
        assert abs(float(v.norm()) - 1.0) > 1e-3, f"layer {layer} looks unit-normalized"
