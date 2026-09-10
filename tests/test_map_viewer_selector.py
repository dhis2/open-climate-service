"""The viewer's selector entries are selection objects, not indices (CLIM-1003).

`indexSelection()` wraps an index as `{selected, type: "index"}` so ZarrLayer selects by array
position rather than resolving the value against a coordinate array it often cannot read. The
wrapping is invisible to Python: the viewer is a template, nothing here executes its JavaScript,
and a slice selected by a NaN index renders as an empty layer rather than an error.

That is not hypothetical. `layerSelector()` was written when a selector entry was a bare index
and clamped it with `Math.min(Math.max(...))`. Merging CLIM-1003 changed the entries to objects
under it; the clamp then produced NaN for every pinned dimension, the branch's own tests passed,
and the template still looked reasonable. These tests pin the convention the code's comments
already claim, so the two halves cannot drift apart again.
"""

import pathlib
import re

_VIEWER = pathlib.Path("open_climate_service/templates/map-viewer.html")
_SOURCE = _VIEWER.read_text(encoding="utf-8")


def test_every_write_to_the_selector_goes_through_index_selection() -> None:
    """A raw index reaching the layer is the fallback CLIM-1003 stopped depending on, and the
    one carbonplan/zarr-layer#87 tightened."""
    writes = re.findall(r"selector\[[^\]]+\]\s*=\s*([^;\n]+)", _SOURCE)
    assert writes, "no selector assignments found; has the viewer been restructured?"
    assert all(w.strip().startswith("indexSelection(") for w in writes), (
        f"a selector entry is assigned something other than indexSelection(...): {writes}"
    )


def test_no_arithmetic_is_performed_on_a_selector_entry() -> None:
    """`Math.min(Math.max(0, selector[k] ?? 0), ...)` on an object is NaN, silently."""
    reads = re.findall(r"Math\.\w+\([^)]*selector\[[^\]]+\](?!\s*\??\.selected)", _SOURCE)
    assert reads == [], f"a selector entry is used as a number: {reads}"


def test_the_pinned_selector_reads_the_selection_and_re_wraps_it() -> None:
    """The positive half: `layerSelector` must unwrap `.selected` and hand the layer a
    selection object back, so pinned dimensions look like every other axis."""
    body = _SOURCE[_SOURCE.index("function layerSelector()") :]
    body = body[: body.index("\n      }")]
    assert "?.selected" in body, "layerSelector no longer unwraps the selection"
    assert "indexSelection(" in body, "layerSelector no longer re-wraps the clamped index"
