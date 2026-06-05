"""L0 / L1 template loading + SHA-256 registration for the manifest.

Researcher-authored text per the Step 4 brief (judgment call carryover
from Step 3 judgment call j — never invent template text):

    agent/data/prompts/L0.txt                    population-wide identity
    scenarios/<scenario_name>/L1_<role>.txt      per-(scenario, role) role

Both files' hashes flow into the run manifest so a replay against a
modified template fails fast rather than silently producing different
results. Step 5's metric pipeline conditions on the L0/L1 hashes.

Loading raises FileNotFoundError with a clear message if the researcher
has not delivered the template yet; this module does not synthesize
template text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LoadedTemplate", "load_L0", "load_L1", "register_in_manifest"]


@dataclass(frozen=True)
class LoadedTemplate:
    """Result of ``load_L0`` / ``load_L1``.

    ``sha256_hex`` is the canonical hash recorded by the manifest so
    a downstream replay against a different template-file body fails
    with a clear error rather than silently shifting prompts.
    """

    path: Path
    text: str
    sha256_hex: str


def load_L0(repo_root: Path) -> LoadedTemplate:
    """Load the population-wide identity template.

    The path is fixed at ``agent/data/prompts/L0.txt`` per the brief's
    deliverable layout. If the file is missing, the error message
    cites the template-spec §1.1 must-not-contain list so the
    researcher knows what to author.
    """
    path = repo_root / "agent" / "data" / "prompts" / "L0.txt"
    return _load(
        path,
        kind="L0",
        guidance=(
            "The researcher authors L0.txt. It must NOT contain any reference to "
            "the simulation as a research experiment, the handler, biases, steering, "
            "the cognitive ledgers, the population-level study, or any specific "
            "scenario's task structure. ~100–150 tokens (Llama 3.1 tokenizer)."
        ),
    )


def load_L1(repo_root: Path, scenario_name: str, role: str) -> LoadedTemplate:
    """Load the per-(scenario, role) template.

    The path is fixed at ``scenarios/<scenario_name>/L1_<role>.txt``.
    """
    path = repo_root / "scenarios" / scenario_name / f"L1_{role}.txt"
    return _load(
        path,
        kind=f"L1[{scenario_name}/{role}]",
        guidance=(
            "The researcher authors L1_<role>.txt for each (scenario, role) pair. It "
            "must NOT contain behavioral preference, any agent's bias, per-tick state, the "
            "agent's specific identity, or experimental manipulations. Describes the "
            "game, never a particular move. ~100–250 tokens."
        ),
    )


def _load(path: Path, *, kind: str, guidance: str) -> LoadedTemplate:
    if not path.exists():
        raise FileNotFoundError(
            f"templates: {kind} template missing at {path}.\n{guidance}"
        )
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LoadedTemplate(path=path, text=text, sha256_hex=sha)


def register_in_manifest(manifest: dict, key: str, tmpl: LoadedTemplate) -> None:
    """Mutate ``manifest`` to record ``key`` → {"path", "sha256"}.

    Used by the agent runtime to write template hashes into the
    run-time manifest before any forward pass executes. Step 5's
    replay machinery compares these hashes against the on-disk
    template body and refuses to run on mismatch.
    """
    manifest.setdefault("templates", {})[key] = {
        "path": str(tmpl.path),
        "sha256": tmpl.sha256_hex,
    }
