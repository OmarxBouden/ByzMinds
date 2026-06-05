"""Tests for the chat-template prompt formatter (Gate 2 root cause #2).

``experiments._vllm_backend.format_probe_prompt`` must route every probe
through the tokenizer's native chat template and never hand-build the
broken literal ``<|system|>`` tags the old formatter used. These tests
use a recording fake tokenizer so they run without transformers/torch.
"""

from __future__ import annotations

from experiments._vllm_backend import format_probe_prompt


class _RecordingTokenizer:
    """Emulates a Llama-3.1-style tokenizer: records the messages it is
    asked to template and emits realistic special tokens so the output
    is observable without loading a model."""

    chat_template = "FAKE"

    def __init__(self):
        self.last = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        parts = ["<|begin_of_text|>"]
        for m in messages:
            parts.append(
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            )
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


def _probe(**kw):
    base = {
        "user_prompt": "State your public position.",
        "output_constraint": "Respond with EXACTLY one of: 'A' or 'B'.",
        "biased_response": "A",
        "honest_response": "B",
    }
    base.update(kw)
    return base


def test_uses_native_special_tokens_not_literal_system_tag():
    tok = _RecordingTokenizer()
    out = format_probe_prompt(_probe(), tok)
    assert "<|begin_of_text|>" in out
    assert "<|start_header_id|>" in out
    assert "<|eot_id|>" in out
    # The old broken format must be gone.
    assert "<|system|>" not in out
    assert "<|user|>" not in out


def test_ends_ready_for_assistant_turn():
    tok = _RecordingTokenizer()
    out = format_probe_prompt(_probe(), tok)
    assert out.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert tok.last["add_generation_prompt"] is True
    assert tok.last["tokenize"] is False


def test_user_turn_includes_constraint():
    tok = _RecordingTokenizer()
    format_probe_prompt(_probe(), tok)
    user_msg = [m for m in tok.last["messages"] if m["role"] == "user"][0]
    assert "State your public position." in user_msg["content"]
    assert "Respond with EXACTLY one of" in user_msg["content"]


def test_system_prompt_placed_as_system_turn_when_present():
    tok = _RecordingTokenizer()
    format_probe_prompt(_probe(system_prompt="You are on a panel."), tok)
    roles = [m["role"] for m in tok.last["messages"]]
    assert roles == ["system", "user"]
    assert tok.last["messages"][0]["content"] == "You are on a panel."


def test_no_system_turn_when_absent():
    tok = _RecordingTokenizer()
    format_probe_prompt(_probe(system_prompt=None), tok)
    roles = [m["role"] for m in tok.last["messages"]]
    assert roles == ["user"]
