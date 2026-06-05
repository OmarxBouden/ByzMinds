"""Smoke test — analysis.metrics module surface is importable."""

from analysis.metrics import (
    bbi,
    distributions,
    divergence,
    manifest_reader,
    surface_honesty,
)


def test_constants_present():
    assert distributions.LAPLACE_ALPHA == 0.5
    assert bbi.DEFAULT_MALFORMATION_POLICY == "drop"
    assert surface_honesty.JUDGE_TOKEN_HONEST == "HONEST"
    assert surface_honesty.JUDGE_TOKEN_DECEPTIVE == "DECEPTIVE"


def test_modules_have_docstrings():
    for m in (bbi, distributions, divergence, manifest_reader, surface_honesty):
        assert m.__doc__, f"{m.__name__} missing docstring"
