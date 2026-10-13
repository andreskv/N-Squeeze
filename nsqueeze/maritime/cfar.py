"""Cell-averaging CFAR ship detector for Sentinel-1 SAR clips.

LNG carriers (≈ 300 m) are bright targets on dark water, so classical constant
false-alarm-rate thresholding works without machine learning. Jetty polygons are
fixed, so clips stay small enough for a laptop.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def ca_cfar(
    intensity: np.ndarray,
    *,
    guard: int = 16,
    background: int = 8,
    pfa: float = 1e-6,
) -> np.ndarray:
    """Boolean detection mask for a 2-D linear-intensity (not dB) SAR image.

    For each pixel, clutter power is the mean over a square ring of half-width
    ``guard + background`` excluding the ``guard`` core. Under exponential clutter
    (single-look intensity), the threshold multiplier for N reference cells is
    α = N · (P_fa^(−1/N) − 1).

    ``guard`` must exceed half the largest target dimension in pixels, or the hull
    leaks into its own clutter estimate and the detection splits (target masking).
    A ≈ 300 m LNG carrier at 10 m pixels is ≈ 30 px long → guard ≥ 16.
    """
    img = np.asarray(intensity, dtype=float)
    outer, inner = 2 * (guard + background) + 1, 2 * guard + 1
    sum_outer = ndimage.uniform_filter(img, outer, mode="reflect") * outer**2
    sum_inner = ndimage.uniform_filter(img, inner, mode="reflect") * inner**2
    n_ref = outer**2 - inner**2
    clutter = (sum_outer - sum_inner) / n_ref
    alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1.0)
    return img > alpha * clutter


def count_vessels(mask: np.ndarray, *, min_pixels: int = 20) -> int:
    """Connected bright blobs of at least ``min_pixels`` (≈ ship-sized at 10 m pixels)."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
    return int(np.sum(sizes >= min_pixels))
