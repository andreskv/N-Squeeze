"""Information. Does the nowcast predict the *surprise* better than baselines?

Reports an information coefficient with HAC (Newey–West) standard errors, because
overlapping weekly/daily observations are autocorrelated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class ICResult:
    ic: float  # standardised slope = correlation
    t_hac: float
    p_hac: float
    n: int
    maxlags: int


def information_coefficient(signal: pd.Series, surprise: pd.Series, *, maxlags: int | None = None):
    """Regress the standardised surprise on the standardised signal with HAC errors."""
    x, y = signal.align(surprise, join="inner")
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        raise ValueError("need at least 10 paired observations")
    lags = maxlags if maxlags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    zx, zy = (x - x.mean()) / x.std(ddof=1), (y - y.mean()) / y.std(ddof=1)
    fit = sm.OLS(zy.to_numpy(), sm.add_constant(zx.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return ICResult(float(fit.params[1]), float(fit.tvalues[1]), float(fit.pvalues[1]), n, lags)


def beats_baselines(errors: dict[str, pd.Series]) -> pd.DataFrame:
    """RMSE table of the model vs naive and seasonal baselines (keys are model names)."""
    rows = {k: float(np.sqrt(np.nanmean(v.to_numpy() ** 2))) for k, v in errors.items()}
    out = pd.Series(rows, name="rmse").to_frame()
    best_baseline = out.drop(index="model", errors="ignore").rmse.min()
    out["relative_to_best_baseline"] = out.rmse / best_baseline
    return out.sort_values("rmse")
