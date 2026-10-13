"""
N-Squeeze: Europe's gas-to-nitrogen chain.

When European gas tightens, who wins and who loses in nitrogen? c
Can physical data see it before prices and company announcements do?

Layers
------
L1  gas_balance      EU storage nowcast and end-of-winter fan chart
L2  cost_stack       C_EU, C_US, urea conversion          (Phase 1)
    import_parity    landed price by origin incl. CBAM    (Phase 1)
    cost_curve       plant merit order and U_pred          (Phase 1)
L3  eaui             satellite plant-utilisation index (HMM)
L4  maritime         SAR berth counts, AIS, chokepoint regime R(t)
    evaluation       the four rigour gates
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nsqueeze")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
