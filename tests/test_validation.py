"""Validation test harness: tolerances are inputs, blocks are explicit."""

import pandas as pd
import pytest

from nsqueeze import validation as v
from nsqueeze.freeze import HoldoutLocked
from nsqueeze.params import ParameterSet


def test_t1_arithmetic_passes_on_graded_h():
    r = v.t1_arithmetic(ParameterSet.load(), rel_tol=0.01)
    assert r.status == "pass" and r.value < 0.01


def test_blocked_and_sealed_tests_say_why():
    assert v.t4_annual_levels().status == "blocked"
    with pytest.raises(HoldoutLocked):
        v.t6_cbam_sanity()


def test_trade_response_detects_lagged_rise():
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    sig = pd.Series(0, index=idx)
    sig.iloc[[10, 25, 40]] = 1
    imp = pd.Series(100.0, index=idx)
    for i in (12, 27, 42):  # imports jump two months after each signal
        imp.iloc[i:] *= 1.5
    tr = v.trade_response(sig, imp).set_index("lag_months")
    assert tr["corr"].idxmax() == 2


def test_t5_placebo_passes_real_lag_and_fails_noise():
    idx = pd.date_range("2015-01-01", periods=120, freq="MS")
    rng = __import__("numpy").random.default_rng(1)
    sig = pd.Series((rng.random(120) < 0.15).astype(int), index=idx)
    imp = pd.Series(100.0, index=idx)
    for i in sig.to_numpy().nonzero()[0]:
        if i + 2 < 120:
            imp.iloc[i + 2 :] *= 1.3  # imports step up two months after each signal
    assert v.t5_trade_response(sig, imp, alpha=0.10, min_offset_months=12).status == "pass"
    # On pure noise the test must pass about alpha of the time, not never and not often.
    passes = [
        v.t5_trade_response(
            sig, pd.Series(rng.random(120) + 1, index=idx), alpha=0.10, min_offset_months=12
        ).status
        == "pass"
        for _ in range(30)
    ]
    assert sum(passes) <= 8  # 30 draws at alpha 0.10: expected 3
