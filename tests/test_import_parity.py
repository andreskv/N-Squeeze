import pytest

from nsqueeze.import_parity import ImportParity, Wedge, cbam_charge, landed_price
from nsqueeze.policy import NoRuleInForce
from nsqueeze.provenance import Grade


@pytest.fixture(scope="module")
def ip():
    return ImportParity()


def test_trinidad_default_cargo_2026_is_about_81_eur(ip):
    """Worked example: (2.444 × 1.01 − 1.522 × 0.975) × €82.32 ≈ €81/t NH3."""
    est = ip.cbam("TT", "2026-08-15")
    assert est.value == pytest.approx(81.0, abs=0.5)


def test_cbam_result_is_flagged_by_placeholder_benchmark(ip):
    est = ip.cbam("TT", "2026-08-15")
    assert est.grade is Grade.C
    assert "cbam.benchmark_ammonia" in est.provenance.flags
    assert "OI-01" in est.provenance.open_items


def test_cbam_price_outside_published_quarter_must_be_supplied(ip):
    with pytest.raises(NoRuleInForce):
        ip.cbam("TT", "2027-03-01")  # no quarterly price recorded yet
    assert ip.cbam("TT", "2027-03-01", cbam_price=80.0).value > 0


def test_cbam_rises_as_free_allocation_phases_out(ip):
    values = [ip.cbam("DZ", f"{y}-06-01", cbam_price=82.32).value for y in range(2026, 2035)]
    assert values == sorted(values)
    # 2034: no free allocation left → whole embedded quantity is charged
    assert values[-1] == pytest.approx(2.101 * 1.01 * 82.32, abs=0.1)


def test_verified_low_emissions_never_go_negative():
    assert cbam_charge(1.0, benchmark=1.522, cbam_factor=0.025, price=80) == 0.0


def test_landed_price_identity():
    w = Wedge(fob=400, freight=50, ad_valorem_duty=450 * 0.055, specific_duty=0, cbam=81)
    assert w.landed == pytest.approx(landed_price(400, 50, ad_valorem=0.055, specific=0, cbam=81))
    assert w.policy_wedge == pytest.approx(24.75 + 81)


def test_mfn_suspension_window(ip):
    before = ip.wedge("SA", "2026-03-01", fob_eur=400, freight_eur=40, cbam_price=80).value
    during = ip.wedge("SA", "2026-08-01", fob_eur=400, freight_eur=40).value
    assert before.ad_valorem_duty == pytest.approx(440 * 0.055)
    assert during.ad_valorem_duty == 0.0


def test_russian_urea_carries_specific_duty(ip):
    w = ip.wedge("RU", "2026-08-01", fob_eur=350, freight_eur=30, product_cn="3102.10").value
    assert w.specific_duty == 60.0
    assert w.ad_valorem_duty == pytest.approx(380 * 0.065)
