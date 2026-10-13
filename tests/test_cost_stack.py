import pytest

from nsqueeze import cost_stack as cs
from nsqueeze.params import ParameterNotFitted, ParameterSet
from nsqueeze.provenance import Grade


@pytest.fixture(scope="module")
def P():
    return ParameterSet.load()


def test_ten_euro_on_ttf_moves_eu_cost_by_about_110():
    g = cs.gas_intensity(35.7)
    c_low = cs.eu_ammonia_cash_cost(30.0, 0.0, g=g, e=0.0)
    c_high = cs.eu_ammonia_cash_cost(40.0, 0.0, g=g, e=0.0)
    assert c_high - c_low == pytest.approx(110, abs=1.0)


@pytest.mark.parametrize(("energy", "e_expected"), [(28.5, 1.6), (35.7, 2.0)])
def test_direct_emissions(energy, e_expected):
    assert cs.emission_intensity(energy) == pytest.approx(e_expected, abs=0.01)


def test_marginal_vs_accounting_kappa():
    # A plant receiving free allocation covering 90 % of emissions has accounting κ = 0.1,
    # but curtailment decisions still use κ = 1.
    e, output = 2.0, 1_000_000
    fa = 0.9 * e * output
    assert cs.accounting_kappa(fa, output, e) == pytest.approx(0.1)


def test_free_allocation_formula():
    assert cs.free_allocation(1.522, 1_000_000, 0.025) == pytest.approx(1_483_950)


def test_cost_gap_rule_of_thumb():
    """Each €1/MWh widening of TTF over HH adds ≈ €11/t to the EU disadvantage (g ≈ 11)."""
    g = cs.gas_intensity(35.7)
    gap_a = cs.eu_ammonia_cash_cost(30, 0, g=g, e=0) - g * 10
    gap_b = cs.eu_ammonia_cash_cost(31, 0, g=g, e=0) - g * 10
    assert gap_b - gap_a == pytest.approx(11.0, abs=0.1)


def test_cost_gap_uses_usd_per_eur():
    assert cs.cost_gap(500.0, 330.0, 1.10) == pytest.approx(200.0)


def test_model_refuses_unfitted_other_costs(P):
    model = cs.CostStack(P)
    tech = cs.PlantTech.tier(P, "eu_average")
    with pytest.raises(ParameterNotFitted):
        model.eu_ammonia(40.0, 80.0, tech)


def test_model_gas_and_carbon_terms_with_provenance(P):
    model = cs.CostStack(P)
    tech = cs.PlantTech.tier(P, "eu_average")
    est = model.eu_ammonia(40.0, 80.0, tech, include_other=False)
    assert est.value == pytest.approx(11.0 * 40 + 2.0 * 80, abs=3)
    assert est.grade is Grade.B  # tier, HHV/LHV and EF are all grade B
    assert not est.flagged


def test_us_cost_is_grade_a(P):
    est = cs.CostStack(P).us_ammonia(3.0, include_other=False)
    assert est.value == pytest.approx(35.3 * 3.0)
    assert est.grade is Grade.A
