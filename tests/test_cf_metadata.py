"""Tests for CF-metadata stamping and units validation (issue #280)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_climate_service.shared.cf import apply_cf_metadata, cf_attrs_from_template, validate_units


def test_cf_attrs_from_template_extracts_cf_fields() -> None:
    template = {
        "units": "mm",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "cell_methods": "time: sum",
        "name": "x",
    }
    assert cf_attrs_from_template(template) == {
        "units": "mm",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "cell_methods": "time: sum",
    }
    assert cf_attrs_from_template(None) == {}
    assert cf_attrs_from_template({"units": ""}) == {"units": ""}  # dimensionless kept


def test_apply_cf_metadata_sets_and_preserves() -> None:
    da = xr.DataArray(np.zeros((2, 2), dtype="float32"), dims=("y", "x"))
    apply_cf_metadata(da, {"units": "mm", "standard_name": "lwe_thickness_of_precipitation_amount"})
    assert da.attrs["units"] == "mm"
    assert da.attrs["standard_name"] == "lwe_thickness_of_precipitation_amount"

    # existing attrs win unless overwrite=True
    da2 = xr.DataArray(np.zeros(2), dims=("x",), attrs={"units": "kg m-2 s-1"})
    apply_cf_metadata(da2, {"units": "mm"})
    assert da2.attrs["units"] == "kg m-2 s-1"
    apply_cf_metadata(da2, {"units": "mm"}, overwrite=True)
    assert da2.attrs["units"] == "mm"


def test_apply_cf_metadata_targets_dataset_variable() -> None:
    ds = xr.Dataset({"precip": (("y", "x"), np.zeros((1, 1), dtype="float32"))}, coords={"y": [0.0], "x": [0.0]})
    apply_cf_metadata(ds, {"units": "mm"}, variable="precip")
    assert ds["precip"].attrs["units"] == "mm"


@pytest.mark.parametrize("units", ["mm", "mm/d", "degC", "kg m-2 s-1", ""])
def test_validate_units_accepts_valid(units: str) -> None:
    assert validate_units(units) is None


@pytest.mark.parametrize("units", ["people", "not-a-unit"])
def test_validate_units_rejects_invalid(units: str) -> None:
    msg = validate_units(units)
    assert msg is not None and "not a recognised" in msg


class TestDropUnserializableAttrs:
    """One guard at the write boundary, rather than a copy inside each plugin that noticed.

    The failure it prevents: Zarr attributes are JSON, so a nested mapping from a source's own
    catalogue writes into the store happily — and then every NetCDF export of that store raises
    ``TypeError: Invalid value for attr``, which surfaces as a 500 from openEO rather than as
    anything wrong with the ingest.
    """

    def _cube(self):  # type: ignore[no-untyped-def]
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {"v": (("t", "y", "x"), np.zeros((1, 2, 2), dtype="float32"))},
            coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]"), "y": [1.0, 0.0], "x": [0.0, 1.0]},
        )

    def test_a_nested_mapping_is_dropped_from_a_coordinate(self) -> None:
        """dynamical.org's GEFS store publishes `statistics_approximate` exactly like this."""
        from open_climate_service.shared.cf import drop_unserializable_attrs

        ds = self._cube()
        ds["x"].attrs["statistics_approximate"] = {"min": -180.0, "max": 179.75}
        ds["x"].attrs["units"] = "degrees_east"

        drop_unserializable_attrs(ds)

        assert "statistics_approximate" not in ds["x"].attrs
        assert ds["x"].attrs["units"] == "degrees_east"

    def test_writeable_attributes_survive(self) -> None:
        """A blocklist of unwriteable types, not an allowlist of known keys: provenance a source
        chose to publish is worth keeping when it can be written."""
        import numpy as np

        from open_climate_service.shared.cf import drop_unserializable_attrs

        ds = self._cube()
        ds["v"].attrs.update(
            {
                "long_name": "a name",
                "count": 3,
                "scale": 1.5,
                "range": [0, 100],
                "pair": (1, 2),
                "array": np.array([1.0, 2.0]),
                "numpy scalar": np.float64(2.5),
                "raw": b"bytes",
                "GRIB_whatever": "provenance we did not ask for but can write",
            }
        )

        drop_unserializable_attrs(ds)

        assert set(ds["v"].attrs) == {
            "long_name",
            "count",
            "scale",
            "range",
            "pair",
            "array",
            "numpy scalar",
            "raw",
            "GRIB_whatever",
        }

    def test_the_types_netcdf_refuses_are_dropped(self) -> None:
        """Derived by trying each against `to_netcdf`, not from the error message it raises —
        which mentions `Number` and so implies bool and complex are fine. They are not."""
        from open_climate_service.shared.cf import drop_unserializable_attrs

        ds = self._cube()
        ds["v"].attrs.update(
            {
                "mapping": {"a": 1},
                "none": None,
                "flag": True,
                "complex": 1 + 2j,
                "nested": [[1, 2], [3, 4]],
                "items": {1, 2},
                "keep": "this one",
            }
        )

        drop_unserializable_attrs(ds)

        assert set(ds["v"].attrs) == {"keep"}

    def test_dataset_level_attributes_are_cleaned_too(self) -> None:
        from open_climate_service.shared.cf import drop_unserializable_attrs

        ds = self._cube()
        ds.attrs.update({"provenance": {"nested": True}, "title": "kept"})

        drop_unserializable_attrs(ds)

        assert set(ds.attrs) == {"title"}

    def test_the_result_actually_writes_to_netcdf(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The point of the guard, asserted against the real writer rather than a type list."""
        import pytest

        from open_climate_service.shared.cf import drop_unserializable_attrs

        ds = self._cube()
        ds["x"].attrs["statistics_approximate"] = {"min": -180.0, "max": 179.75}
        with pytest.raises(TypeError):
            ds.to_netcdf(tmp_path / "before.nc")

        drop_unserializable_attrs(ds)
        ds.to_netcdf(tmp_path / "after.nc")
        assert (tmp_path / "after.nc").exists()


def test_netcdf_export_survives_a_store_carrying_geozarr_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An OCS store's own root attributes are not NetCDF-writeable, by design.

    ``write_geozarr_attrs`` declares the GeoZarr convention as ``zarr_conventions``, a list of
    mappings. That is correct Zarr metadata with no NetCDF equivalent, so the store keeps it and
    the export drops it — which is why the guard runs on the export path as well as at ingest.
    """
    import numpy as np
    import xarray as xr

    from open_climate_service.openeo.jobs import _write_raster

    ds = xr.Dataset(
        {"v": (("t", "y", "x"), np.zeros((1, 2, 2), dtype="float32"))},
        coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]"), "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    ds.attrs["zarr_conventions"] = [{"uuid": "689b58e2", "schema_url": "https://example.invalid/geozarr"}]
    ds.attrs["spatial:transform"] = [1.0, 0.0, 33.0, 0.0, -1.0, -9.0]

    path = _write_raster(ds, tmp_path, "NETCDF")

    assert path is not None
    written = xr.open_dataset(path)
    try:
        assert "zarr_conventions" not in written.attrs
        # A flat list of numbers is writeable, so it is kept rather than blanket-stripped.
        assert list(written.attrs["spatial:transform"]) == [1.0, 0.0, 33.0, 0.0, -1.0, -9.0]
    finally:
        written.close()
