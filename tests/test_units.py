"""Units and the day-1 arithmetic check."""

import pytest

from nsqueeze import units as u


@pytest.mark.parametrize(("h", "expected"), [(35.3, 32.0), (36.4, 33.0)])
def test_h_reproduces_cf_sensitivity_per_short_ton(h, expected):
    """h (MMBtu/t) must reproduce CF's disclosed $32–33 per short ton per $1/MMBtu."""
    usd_per_short_ton = u.per_t_to_per_short_ton(h * 1.0)
    assert usd_per_short_ton == pytest.approx(expected, abs=0.05)


def test_inverting_cf_sensitivity_gives_h():
    assert u.gas_use_from_sensitivity(32) == pytest.approx(35.27, abs=0.01)
    assert u.gas_use_from_sensitivity(33) == pytest.approx(36.38, abs=0.01)


def test_icis_cross_check_is_close():
    """ICIS independently uses ≈ 33 MMBtu/t; CF-implied h must sit within ~10 %."""
    assert abs(u.gas_use_from_sensitivity(32) - 33) / 33 < 0.10


def test_energy_definitions():
    assert u.MMBTU_PER_MWH == pytest.approx(3.412, abs=1e-3)
    assert u.mmbtu_to_mwh(u.mwh_to_mmbtu(7.5)) == pytest.approx(7.5)


def test_lhv_intensity_to_gas_purchased():
    # EU average 35.7 GJ LHV/t → ≈ 11.0 MWh HHV/t
    assert u.gj_lhv_to_mwh_hhv(35.7) == pytest.approx(11.0, abs=0.05)
    assert u.gj_lhv_to_mwh_hhv(28.5) == pytest.approx(8.77, abs=0.02)
    assert u.gj_lhv_to_mwh_hhv(29.7) == pytest.approx(9.14, abs=0.02)


def test_price_conversion_round_trip():
    ttf_eur_mwh, fx = 40.0, 1.10
    usd_mmbtu = u.eur_mwh_to_usd_mmbtu(ttf_eur_mwh, fx)
    assert usd_mmbtu == pytest.approx(12.90, abs=0.01)
    assert u.usd_mmbtu_to_eur_mwh(usd_mmbtu, fx) == pytest.approx(ttf_eur_mwh)


def test_stoichiometry():
    assert u.NH3_PER_T_UREA == pytest.approx(0.567, abs=5e-4)
    assert u.CO2_BOUND_PER_T_UREA == pytest.approx(0.733, abs=5e-4)
    assert u.NH3_PER_T_N == pytest.approx(1.216, abs=5e-4)
    assert u.N_CONTENT_NH3 == pytest.approx(0.822, abs=5e-4)
    assert u.N_CONTENT_UREA == pytest.approx(0.466, abs=5e-4)


def test_urea_conversion_step_derivation():
    """Inference: urea's own step ≈ 24.3 − 0.567 × 35.3 ≈ 4.3 MMBtu/t."""
    urea_total = u.gas_use_from_sensitivity(22)  # CF: $22/short ton for granular urea
    assert urea_total == pytest.approx(24.3, abs=0.05)
    assert urea_total - u.NH3_PER_T_UREA * 35.3 == pytest.approx(4.3, abs=0.1)
