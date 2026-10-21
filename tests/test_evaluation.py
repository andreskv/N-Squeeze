import numpy as np
import pandas as pd
import pytest

from nsqueeze.evaluation.conditions import Condition, ConditionResult, SignalReport
from nsqueeze.evaluation.measurement import change_snr
from nsqueeze.evaluation.nulls import block_bootstrap_ci, circular_shift_placebo
from nsqueeze.evaluation.tradability import deflated_sharpe, expected_max_sharpe
from nsqueeze.maritime.cfar import ca_cfar, count_vessels
from nsqueeze.plant_activity import plant_activity_index


def test_snr_punishes_level_trackers_with_noisy_changes():
    """The Cushing lesson: good level tracking, useless changes."""
    rng = np.random.default_rng(0)
    t = np.arange(200)
    seasonal = 60 * np.sin(2 * np.pi * t / 52)  # a big, slow inventory cycle
    truth = pd.Series(400 + seasonal + np.cumsum(rng.normal(0, 1.5, 200)))
    noisy = truth + rng.normal(0, 15, 200)  # level error ≫ weekly change
    good = truth + rng.normal(0, 0.3, 200)
    assert noisy.corr(truth) > 0.85  # looks fine on levels…
    assert change_snr(noisy, truth) < 1  # …but fails condition 1 (measurement)
    assert change_snr(good, truth) > 2


def test_more_trials_raise_the_hurdle():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0


def test_deflated_sharpe_falls_with_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0008, 0.01, 750)
    assert deflated_sharpe(r, n_trials=1, trial_sr_variance=0.002) > deflated_sharpe(
        r, n_trials=200, trial_sr_variance=0.002
    )


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_circular_shift_detects_real_signal_and_not_noise():
    rng = np.random.default_rng(2)
    t = 500
    signal = np.convolve(rng.normal(size=t + 9), np.ones(10) / 10, "valid")  # autocorrelated
    outcome = 0.6 * signal + rng.normal(0, 0.3, t)
    real = circular_shift_placebo(signal, outcome, _corr, n=500, min_offset=30, rng=3)
    fake = circular_shift_placebo(signal, rng.normal(size=t), _corr, n=500, min_offset=30, rng=3)
    assert real.p_value < 0.01
    assert fake.p_value > 0.05


def test_block_bootstrap_interval_contains_mean():
    x = np.random.default_rng(4).normal(1.0, 1.0, 400)
    lo, hi = block_bootstrap_ci(x, np.mean, block=20, n=500, rng=5)
    assert lo < 1.0 < hi


def test_gates_must_run_in_order_and_stop_on_failure():
    rep = SignalReport("Unexpected curtailment")
    rep.record(ConditionResult(Condition.MEASUREMENT, False, "snr", 0.8, 1.0))
    assert rep.stopped_at is Condition.MEASUREMENT
    with pytest.raises(ValueError):
        rep.record(ConditionResult(Condition.INFORMATION, True, "ic", 0.2, 0.1))


def test_cfar_finds_a_ship_on_sea_clutter():
    rng = np.random.default_rng(6)
    sea = rng.exponential(1.0, (200, 200))
    sea[90:96, 60:90] += 400.0  # a ~300 m bright hull at 10 m pixels
    assert count_vessels(ca_cfar(sea), min_pixels=20) == 1


def test_cfar_guard_smaller_than_hull_splits_the_target():
    """Target masking: documents why the default guard is sized to an LNG carrier."""
    rng = np.random.default_rng(6)
    sea = rng.exponential(1.0, (200, 200))
    sea[90:96, 60:90] += 400.0
    assert count_vessels(ca_cfar(sea, guard=6, background=10), min_pixels=20) == 2


def test_plant_activity_index_weights_by_capacity_and_ignores_missing():
    p = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, np.nan]})
    cap = pd.Series({"A": 300.0, "B": 100.0})
    assert plant_activity_index(p, cap).tolist() == pytest.approx([0.75, 1.0])


def test_label_weeks_trims_off_periods_and_drops_conflicts():
    from nsqueeze.plant_activity.weekly import label_weeks

    weeks = pd.date_range("2022-01-03", periods=10, freq="W-MON")
    labels = pd.DataFrame(
        {
            "plant_id": ["A", "B", "B"],
            "start": ["2022-01-03", "2022-01-03", "2022-01-03"],
            "end": ["2022-03-13", "2022-01-16", "2022-01-16"],
            "label": ["off", "running", "off"],
        }
    )
    out = label_weeks(labels, weeks)
    a = out[out.plant_id == "A"].week
    assert a.min() == pd.Timestamp("2022-01-17") and a.max() == pd.Timestamp("2022-02-21")
    assert (out[out.plant_id == "B"].label == "running").all()  # B's off period trims to nothing


def test_rank_auc():
    from nsqueeze.plant_activity.weekly import auc

    off = pd.Series([True, True, False, False])
    assert auc(pd.Series([3.0, 4.0, 1.0, 2.0]), off) == 1.0
    assert auc(pd.Series([1.0, 1.0, 1.0, 1.0]), off) == 0.5
