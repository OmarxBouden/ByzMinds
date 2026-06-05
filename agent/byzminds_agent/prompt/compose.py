"""Chat-template assembly per byzminds-template-spec.md §3.1.

The canonical form is dual-system:

    [
      {"role": "system",  "content": L0},
      {"role": "system",  "content": L1},
      {"role": "user",    "content": L2},
    ]

Llama 3.1's chat template accepts multiple system messages. Experiment
011 measures whether the model adheres to L0+L1 instructions more
cleanly under dual-system vs concatenated single-system, and the
choice is locked for all Stage A. The fallback (concat) joins L0 + L1
with an explicit separator that the model is likely to treat as a
section boundary:

    L0 + "\\n\\n---\\n\\n" + L1

The locked mode is recorded in the run manifest (compose returns it
alongside the messages list).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ChatComposition", "compose_chat_input", "SYSTEM_CONCAT_SEPARATOR"]

SYSTEM_CONCAT_SEPARATOR = "\n\n---\n\n"

Mode = Literal["dual_system", "concat_single_system"]


@dataclass(frozen=True)
class ChatComposition:
    """Returned from :func:`compose_chat_input`.

    Fields go into the manifest so a reviewer can reproduce the
    exact prompt that produced the run.
    """

    messages: list[dict[str, str]]
    tool_schemas: list[dict]
    mode: Mode


def compose_chat_input(
    L0_text: str,
    L1_text: str,
    L2_text: str,
    tool_schemas: list[dict],
    *,
    mode: Mode = "dual_system",
) -> ChatComposition:
    """Build the chat-template input for the model.

    Parameters
    ----------
    L0_text, L1_text, L2_text
        Template-spec layers 0/1/2. Already-rendered strings.
    tool_schemas
        OpenAI-style tool schemas list; passed through unmodified.
    mode
        ``"dual_system"`` (default) or ``"concat_single_system"`` —
        locked once per run from Experiment 011's outcome.
    """
    if mode == "dual_system":
        messages = [
            {"role": "system", "content": L0_text},
            {"role": "system", "content": L1_text},
            {"role": "user", "content": L2_text},
        ]
    elif mode == "concat_single_system":
        messages = [
            {"role": "system", "content": L0_text + SYSTEM_CONCAT_SEPARATOR + L1_text},
            {"role": "user", "content": L2_text},
        ]
    else:
        raise ValueError(f"unknown compose mode: {mode!r}")
    return ChatComposition(messages=messages, tool_schemas=tool_schemas, mode=mode)
