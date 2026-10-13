"""Layer 3 · European Ammonia Utilisation Index.

Per plant and per week, the probability that each ammonia plant is running, from
several noisy satellite sensors fused in a hidden Markov model.

    EAUI(t) = Σ cap_i · P(running_i, t) / Σ cap_i

Kill criterion: out-of-sample AUC on held-out episodes < 0.75 → publish
as a negative result and restrict the EAUI to the large sites that clear the bar.
"""

from nsqueeze.eaui.index import eaui
from nsqueeze.eaui.sensors import SENSORS, Sensor

__all__ = ["SENSORS", "Sensor", "eaui"]
