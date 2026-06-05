"""theta_for_persona + DIALS-order pinning (the L_cog_ind disposition label).

The DIALS order is the canonical length-6 theta index order recorded on
L_cog_ind; the Go kernel's byzminds-panel dialOrder MUST mirror it (asserted on
the Go side by TestDialOrderMatchesPython). This test pins the Python side.
"""

from byzminds_agent import DIALS, STRENGTH_MAGNITUDE, theta_for_persona


def test_dials_order_pinned():
    assert DIALS == ("authority", "bandwagon", "sycophancy", "free_ride", "collude", "deceive")


def test_theta_for_persona_collude_strong():
    th = theta_for_persona("collude", "strong")
    assert len(th) == 6
    assert th[DIALS.index("collude")] == 1.0
    assert sum(1 for x in th if x != 0) == 1  # one-hot at the dial index


def test_theta_for_persona_strengths():
    assert theta_for_persona("authority", "moderate")[0] == STRENGTH_MAGNITUDE["moderate"]
    assert theta_for_persona("authority", "mild")[0] == STRENGTH_MAGNITUDE["mild"]


def test_theta_for_persona_honest_is_zero():
    assert theta_for_persona("", "none") == [0.0] * 6
    assert theta_for_persona("collude", "none") == [0.0] * 6


def test_theta_for_persona_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        theta_for_persona("nonsense", "strong")
