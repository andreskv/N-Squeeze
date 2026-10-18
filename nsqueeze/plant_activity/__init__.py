"""Plant activity index: how much of Europe's ammonia capacity is running.

Per plant and per week, the probability that each ammonia plant is running, from
several noisy satellite sensors fused in a hidden Markov model.

    index(t) = Σ cap_i · P(running_i, t) / Σ cap_i

Satellite detection test: out-of-sample AUC on held-out episodes < 0.75 → publish
as a negative result and restrict the index to the large sites that clear the bar.
"""

from nsqueeze.plant_activity.index import plant_activity_index
from nsqueeze.plant_activity.sensors import SENSORS, Sensor

__all__ = ["SENSORS", "Sensor", "plant_activity_index"]
