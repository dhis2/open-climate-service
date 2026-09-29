"""Class data: which variables are categorical, the weighted majority, and per-class fractions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr

from open_climate_service.shared.vectors import RESAMPLING_ATTR

CATEGORICAL_RESAMPLING = frozenset({"mode", "max", "nearest"})

# Numeric reducers averaging class codes would be meaningless; on categorical data they become
# a majority. min and max keep their meaning (max over a presence mask is "any present").
CATEGORICAL_REPLACED = frozenset({"mean", "median", "sum"})

FRACTIONS_DIM = "class"
"""Dimension a ``fractions`` aggregation adds: one entry per class value found in the zones."""


def weighted_majority(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The class covering the most area; ties go to the smallest class value."""
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full(flat_v.shape[0], np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        keep = w > 0
        if not keep.any():
            continue
        classes, inverse = np.unique(v[keep], return_inverse=True)
        out[i] = classes[np.argmax(np.bincount(inverse, weights=w[keep]))]
    return out.reshape(values.shape[:-1])


MAX_FRACTION_CLASSES = 256
"""Most distinct values ``fractions`` will return shares for.

Land-cover schemes have tens of classes. A continuous layer has as many values as cells, which
would make the output as large as the input; that is refused rather than computed.
"""


def weighted_fractions(values: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Share of each zone's valid area in each class, as (…, class).

    One pass over the cells: each is binned by its class index, so the work does not grow
    with the number of classes. A zone whose covered cells are all missing is NaN in every class.
    """
    shape = values.shape[:-1]
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full((flat_v.shape[0], classes.size), np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        valid = ~np.isnan(v) & (w > 0)
        total = w[valid].sum()
        if not classes.size or total <= 0:
            continue
        index = np.searchsorted(classes, v[valid])
        out[i] = np.bincount(index, weights=w[valid], minlength=classes.size) / total
    return out.reshape(*shape, classes.size)


def fraction_classes(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """The class axis shared by every variable, so each variable's shares line up.

    Merged one array at a time and refused as soon as it passes the limit, so a continuous
    layer fails after one block rather than after every block's values have been collected.
    """
    classes = np.array([], dtype="float64")
    for a in arrays:
        classes = np.union1d(classes, a[~np.isnan(a)])
        if classes.size > MAX_FRACTION_CLASSES:
            raise ValueError(
                f"aggregate_spatial: fractions found more than {MAX_FRACTION_CLASSES} distinct values in "
                "the zones; fractions are for class codes, not continuous data"
            )
    return classes


def categorical_variables(data: xr.Dataset) -> set[str]:
    """The variables whose dataset declares a categorical ``ingestion.resampling``.

    Decided per variable, from the marker ``load_collection`` sets on the variable itself, so a
    cube merging land cover with temperature averages the temperature and takes the majority of
    the land cover. A cube-level attribute is not consulted: merging copies one variable's
    attributes up to the cube, which would mark every variable as categorical.
    """
    return {
        str(name)
        for name, da in data.data_vars.items()
        if str(da.attrs.get(RESAMPLING_ATTR, "")) in CATEGORICAL_RESAMPLING
    }
