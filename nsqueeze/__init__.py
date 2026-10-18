"""
N-Squeeze: Europe's gas-to-nitrogen chain.

When European gas tightens, who wins and who loses in nitrogen?
Can physical data see it before prices and company announcements do?

Models
------
Gas storage model     gas_balance      EU storage nowcast and end-of-winter fan chart
Nitrogen cost model   cost_stack       C_EU, C_US, urea conversion
                      import_parity    landed price by origin incl. CBAM
                      cost_curve       plant merit order and U_pred
Plant activity index  plant_activity   satellite plant-utilisation index (HMM)
Shipping and trade    maritime         SAR berth counts, AIS, chokepoint regime R(t)
                      evaluation       the four conditions for a signal
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nsqueeze")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
