"""Built-in dataset transform functions.

Each function has the signature:
    (ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset
"""

from .unit_conversion import (
    kelvin_difference_to_celsius,
    kelvin_to_celsius,
    metres_per_second_to_mm_per_day,
    metres_to_mm,
)

__all__ = [
    "kelvin_difference_to_celsius",
    "kelvin_to_celsius",
    "metres_per_second_to_mm_per_day",
    "metres_to_mm",
]
