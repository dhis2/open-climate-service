"""Categorical datasets: class codes with labels and colours (CLIM-895).

A dataset template declares its classes, and with them that its values are codes rather than
measurements: land cover, hazard and warning levels, binary masks. Averaging codes gives a
plausible but wrong class, so a categorical dataset is aggregated per class instead.

```yaml
classes:
  - { value: 0, label: No warning, color: "#ffffb2" }
  - { value: 1, label: Caution (35-38 °C), color: "#fecc5c" }
display:
  nodata: 255   # optional: a code that is not data, so 0 can be a class
```

The classes reach the store as CF ``flag_values``/``flag_meanings`` on the variable, STAC as
``classification:classes``, and a loaded cube as the same CF attributes, stamped from the
template at load so a store written before its classes were declared is still categorical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STAC_CLASSIFICATION_EXTENSION = "https://stac-extensions.github.io/classification/v2.0.0/schema.json"

# Stamped on a loaded cube beside the CF flags, so a refusal can name the dataset.
DATASET_ATTR = "ocs_dataset"
NODATA_ATTR = "ocs_nodata"

_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")


@dataclass(frozen=True)
class DataClass:
    """One class: its code in the data, a label for people, and an optional display colour."""

    value: int
    label: str
    color: str | None = None


@dataclass(frozen=True)
class Classification:
    """The classes a categorical dataset declares, and the code that is not data, if any."""

    classes: tuple[DataClass, ...]
    nodata: int | None = None

    @property
    def values(self) -> list[int]:
        return [c.value for c in self.classes]

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.classes]


def classification_from_template(template: dict[str, Any] | None) -> Classification | None:
    """The classes a template declares, or None for a continuous dataset.

    Raises ValueError for a malformed declaration, naming what is wrong, so a template that
    would publish wrong class metadata is refused when it is loaded.
    """
    raw = (template or {}).get("classes")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("'classes' must be a non-empty list of {value, label, color}")
    classes: list[DataClass] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"classes[{index}] must be a mapping with 'value' and 'label'")
        value, label, color = entry.get("value"), entry.get("label"), entry.get("color")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"classes[{index}].value must be an integer code, got {value!r}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"classes[{index}].label must be a non-empty string")
        if color is not None and (not isinstance(color, str) or not _COLOR.fullmatch(color)):
            raise ValueError(f"classes[{index}].color must be a hex colour such as '#1a9850', got {color!r}")
        classes.append(DataClass(value=value, label=label.strip(), color=color))
    values = [c.value for c in classes]
    if len(set(values)) != len(values):
        raise ValueError(f"'classes' declares a value more than once: {values}")

    display = (template or {}).get("display")
    if display is not None and not isinstance(display, dict):
        raise ValueError(f"'display' must be a mapping, got {type(display).__name__}")
    nodata_raw = (display or {}).get("nodata")
    nodata: int | None = None
    if nodata_raw is not None:
        integral = isinstance(nodata_raw, int) or (isinstance(nodata_raw, float) and nodata_raw.is_integer())
        if isinstance(nodata_raw, bool) or not integral:
            raise ValueError(f"display.nodata of a categorical dataset must be an integer code, got {nodata_raw!r}")
        nodata = int(nodata_raw)
        if nodata in values:
            raise ValueError(f"display.nodata {nodata} is also declared as a class")
    return Classification(classes=tuple(classes), nodata=nodata)


def cf_flag_attrs(classification: Classification) -> dict[str, Any]:
    """CF ``flag_values`` and ``flag_meanings``: the codes, and one word per label."""
    meanings = " ".join(re.sub(r"[^0-9A-Za-z_.+@-]+", "_", label).strip("_") for label in classification.labels)
    return {"flag_values": classification.values, "flag_meanings": meanings}


def stac_classes(classification: Classification) -> list[dict[str, Any]]:
    """STAC ``classification:classes``: one entry per class, and the nodata code marked as such."""
    entries: list[dict[str, Any]] = []
    for item in classification.classes:
        entry: dict[str, Any] = {
            "value": item.value,
            "name": re.sub(r"[^0-9A-Za-z_-]+", "_", item.label).strip("_") or f"class_{item.value}",
            "description": item.label,
        }
        if item.color:
            entry["color_hint"] = item.color.lstrip("#").upper()
        entries.append(entry)
    if classification.nodata is not None:
        entries.append({"value": classification.nodata, "name": "nodata", "nodata": True})
    return entries


def stamp_classes(data: Any, collection_id: str, classification: Classification | None) -> Any:
    """Mark a loaded cube with its dataset's classes, in place, and return it."""
    attrs = getattr(data, "attrs", None)
    if isinstance(attrs, dict) and classification is not None:
        attrs.update(cf_flag_attrs(classification))
        attrs[DATASET_ATTR] = collection_id
        if classification.nodata is not None:
            attrs[NODATA_ATTR] = classification.nodata
    return data


def cube_classes(data: Any) -> tuple[list[int], list[str]] | None:
    """The class codes and labels a cube carries as CF flags, or None for a continuous cube."""
    attrs = getattr(data, "attrs", None) or {}
    values = attrs.get("flag_values")
    if values is None:
        return None
    codes = [int(v) for v in (values.tolist() if hasattr(values, "tolist") else list(values))]
    meanings = str(attrs.get("flag_meanings", "")).split()
    labels = meanings if len(meanings) == len(codes) else [str(code) for code in codes]
    return codes, labels


def cube_name(data: Any) -> str:
    """The dataset a cube was loaded from, for messages, or its variable name."""
    attrs = getattr(data, "attrs", None) or {}
    return str(attrs.get(DATASET_ATTR) or getattr(data, "name", None) or "the input")
