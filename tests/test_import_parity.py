import pytest

from nsqueeze.import_parity import ImportParity, Wedge, cbam_charge, landed_price
from nsqueeze.policy import NoRuleInForce


@pytest.fixture(scope="module")
def ip():
    return ImportParity()


def test_trinidad_default_cargo_2026_is_about_79_eur(ip):
    """Worked example: (2.420 × 1.01 − 1.522 × 0.975) × €82.32 ≈ €79/t NH3."""
    est = ip.cbam("TT", "2026-08-15")
    assert est.value == pytest.approx(79.0, abs=0.5)


def test_cbam_inputs_are_official_values(ip):
    est = ip.cbam("TT", "2026-08-15")
    assert "cbam.benchmark_ammonia" not in est.provenance.flags
    assert "OI-01" not in est.provenance.open_items


def test_origin_without_a_country_row_uses_other_countries(ip):
    # CN: no row on file -> "Other countries and territories" (3.130)
    est = ip.cbam("CN", "2026-08-15")
    assert est.value == pytest.approx((3.130 * 1.01 - 1.522 * 0.975) * 82.32, abs=0.1)


def test_cbam_price_outside_published_quarter_must_be_supplied(ip):
    with pytest.raises(NoRuleInForce):
        ip.cbam("TT", "2027-03-01")  # no quarterly price recorded yet
    assert ip.cbam("TT", "2027-03-01", cbam_price=80.0).value > 0


def test_cbam_rises_as_free_allocation_phases_out(ip):
    # No CSCF decision exists after 2030, so the caller supplies one (here 100 %).
    values = [
        ip.cbam("DZ", f"{y}-06-01", cbam_price=82.32, cscf=1.0).value for y in range(2026, 2035)
    ]
    assert values == sorted(values)
    # 2034: no free allocation left → whole embedded quantity is charged
    assert values[-1] == pytest.approx(2.080 * 1.01 * 82.32, abs=0.1)


def test_cscf_is_read_from_the_schedule_and_required_after_2030(ip):
    est = ip.cbam("TT", "2026-08-15")
    assert any(r.key.startswith("policy:cscf[EU]") for r in est.provenance.refs)
    with pytest.raises(NoRuleInForce):
        ip.cbam("TT", "2031-06-01", cbam_price=80.0)
    lower = ip.cbam("TT", "2026-08-15", cscf=0.9).value
    assert lower == pytest.approx((2.420 * 1.01 - 1.522 * 0.975 * 0.9) * 82.32, abs=0.1)


def test_urea_cargo_uses_urea_default_and_benchmark(ip):
    est = ip.cbam("TT", "2026-08-15", product_cn="3102.10")
    assert est.value == pytest.approx((1.610 * 1.01 - 0.902 * 0.975) * 82.32, abs=0.1)
    w = ip.wedge("TT", "2026-08-15", fob_eur=300, freight_eur=30, product_cn="3102.10")
    assert w.value.cbam == pytest.approx(est.value)


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
