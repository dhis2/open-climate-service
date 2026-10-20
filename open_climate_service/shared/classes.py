"""Categorical datasets: class codes with labels and colours (CLIM-895).

A dataset template declares its classes, and with them that its values are codes rather than
measurements: land cover, hazard and warning levels, binary masks. Averaging codes gives a
plausible but wrong class, so a categorical dataset is aggregated per class instead.

```yaml
classes:
  - { value: 0, label: No warning, color: "#ffffb2" }
  - { value: 1, label: Caution (35-38 °C), color: "#fecc5c" }
```

The classes reach the store as CF ``flag_values``/``flag_meanings`` on the variable, which a
loaded cube carries, and STAC as ``classification:classes``. A cell that is no class is stored
as missing (NaN) by its plugin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STAC_CLASSIFICATION_EXTENSION = "https://stac-extensions.github.io/classification/v2.0.0/schema.json"

_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")

# ColorBrewer Set3: a categorical palette, so neighbouring codes are told apart rather than shaded.
_FALLBACK_PALETTE = (
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
)


@dataclass(frozen=True)
class DataClass:
    """One class: its code in the data, a label for people, and an optional display colour."""

    value: int
    label: str
    color: str | None = None


@dataclass(frozen=True)
class Classification:
    """The classes a categorical dataset declares."""

    classes: tuple[DataClass, ...]

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
    return Classification(classes=tuple(classes))


def class_colors(classification: Classification) -> list[str]:
    """Each class's display colour: its declared one, else the next of the fallback palette.

    The map viewer falls back to the same palette in the same order, so a class without a
    declared colour looks the same on the map, in a thumbnail and in a page's legend.
    """
    return [
        item.color or _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]
        for index, item in enumerate(classification.classes)
    ]


def cf_flag_attrs(classification: Classification) -> dict[str, Any]:
    """CF ``flag_values`` and ``flag_meanings``: the codes, and one word per label."""
    meanings = " ".join(re.sub(r"[^0-9A-Za-z_.+@-]+", "_", label).strip("_") for label in classification.labels)
    return {"flag_values": classification.values, "flag_meanings": meanings}


def stac_classes(classification: Classification) -> list[dict[str, Any]]:
    """STAC ``classification:classes``: one entry per class."""
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
    return entries


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
