"""Built-in dataset transform functions.

Each function has the signature:
    (ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset
"""

from .unit_conversion import kelvin_to_celsius, metres_to_mm

__all__ = ["kelvin_to_celsius", "metres_to_mm"]
