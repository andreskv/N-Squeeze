import pandas as pd
import pytest

from nsqueeze.freeze import FreezeState, HoldoutLocked, load_episodes, require_frozen
from nsqueeze.params import ParameterNotFitted, ParameterSet
from nsqueeze.policy import NoRuleInForce, PolicySchedule
from nsqueeze.provenance import Grade


@pytest.fixture(scope="module")
def pol():
    return PolicySchedule.load()


def test_cbam_factor_schedule(pol):
    assert pol.value("cbam_factor", "EU", "2026-05-01").value == 0.025
    assert pol.value("cbam_factor", "EU", "2030-01-01").value == 0.485
    assert pol.value("cbam_factor", "EU", "2040-01-01").value == 1.0
    with pytest.raises(NoRuleInForce):
        pol.value("cbam_factor", "EU", "2025-06-01")


def test_point_in_time_benchmark(pol):
    assert pol.value("ets_benchmark", "NH3", "2024-01-01").value == 1.57
    assert pol.value("ets_benchmark", "NH3", "2026-01-01").value == 1.522


def test_unresolved_rule_raises_with_open_item(tmp_path):
    f = tmp_path / "schedule.csv"
    f.write_text(
        "parameter,scope,value,unit,effective_from,effective_to,grade,source,notes\n"
        "duty_specific,X:2814,,EUR/t,2025-07-01,,C,-,OI-99 not known yet\n"
    )
    with pytest.raises(NoRuleInForce, match="OI-99"):
        PolicySchedule.load(f).value("duty_specific", "X:2814", "2026-08-01")


def test_russian_ammonia_has_no_specific_duty(pol):
    assert pol.value("duty_specific", "RU_BY:2814", "2026-08-01").value == 0


def test_series_lookup(pol):
    s = pol.series("cbam_factor", "EU", pd.date_range("2025-12-30", periods=4, freq="D"))
    assert s.isna().sum() == 2 and (s.dropna() == 0.025).all()


def test_parameter_registry():
    P = ParameterSet.load()
    assert P.value("ammonia_tech.energy_eu_average") == 35.7
    assert P["cbam.default_embedded_emissions.TT"].source.startswith(
        "IR (EU) 2025/2621"
    )  # inherited
    assert P["cbam.benchmark_ammonia"].grade is Grade.A
    unfitted = {p.key for p in P.unfitted()}
    assert {"fitted.run_threshold_theta", "fitted.other_cash_cost_eu"} <= unfitted
    with pytest.raises(ParameterNotFitted):
        P.value("fitted.run_softness_s")


def test_holdout_is_locked_until_frozen():
    with pytest.raises(HoldoutLocked):
        require_frozen(FreezeState(frozen=False, preregistration_commit=None, frozen_at=None))
    with pytest.raises(HoldoutLocked):
        load_episodes(include_holdout=True)
    calib = load_episodes()
    assert (calib["date"] < "2026-01-01").all()
