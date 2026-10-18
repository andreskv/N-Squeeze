"""Rigour protocol: measurement → information → price impact → tradability."""

from nsqueeze.evaluation.conditions import Condition, ConditionResult, SignalReport
from nsqueeze.evaluation.measurement import change_snr
from nsqueeze.evaluation.nulls import block_bootstrap_ci, circular_shift_placebo
from nsqueeze.evaluation.tradability import deflated_sharpe

__all__ = [
    "Condition",
    "ConditionResult",
    "SignalReport",
    "block_bootstrap_ci",
    "change_snr",
    "circular_shift_placebo",
    "deflated_sharpe",
]
