"""Smoke test — package imports + DIALS constant is the expected six."""

from byzminds_agent import DIALS


def test_dials_constant() -> None:
    assert DIALS == (
        "authority",
        "bandwagon",
        "sycophancy",
        "free_ride",
        "collude",
        "deceive",
    )


def test_proto_gen_package_importable() -> None:
    import byzminds_agent.proto_gen  # noqa: F401
