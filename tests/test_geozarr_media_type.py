"""Media type advertised for published GeoZarr stores (flat vs pyramided).

The `profile=multiscales` parameter is what gates rendering in pyramid-only clients. It is
matched as a literal — stac-js compares against
`['application/vnd.zarr; version=3; profile=multiscales']` with
`allowedTypes.includes(type.toLowerCase())` — so case is forgiven but parameter order and
spacing are not, and any reformatting silently disables rendering everywhere (CLIM-853).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from open_geo_service.stac.media_types import (
    MULTISCALES_CONVENTION_UUID,
    WEB_OPTIMIZED_ZARR_MEDIA_TYPE,
    ZARR_V3_MEDIA_TYPE,
    attributes_declare_multiscales,
    read_root_attributes,
    zarr_media_type,
)


def _multiscales_attrs(levels: int = 3) -> dict[str, Any]:
    """Root attributes as geozarr_toolkit writes them for a pyramided store."""
    return {
        "zarr_conventions": [
            {"name": "spatial:", "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"},
            {"name": "proj:", "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f"},
            {"name": "multiscales", "uuid": MULTISCALES_CONVENTION_UUID},
        ],
        "multiscales": {
            "layout": [{"path": str(level)} for level in range(levels)],
            "resampling_method": "mean",
        },
        "proj:code": "EPSG:4326",
        "spatial:bbox": [0.0, 0.0, 1.0, 1.0],
    }


def _write_zarr_store(root: Path, attributes: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes}),
        encoding="utf-8",
    )
    return root


def test_web_optimized_media_type_is_byte_identical_to_client_expectation() -> None:
    # Consumers match this string literally (no media type parameter parsing),
    # so any reformatting silently disables pyramid rendering everywhere.
    assert WEB_OPTIMIZED_ZARR_MEDIA_TYPE == "application/vnd.zarr; version=3; profile=multiscales"
    assert ZARR_V3_MEDIA_TYPE == "application/vnd.zarr; version=3"


class TestAttributesDeclareMultiscales:
    def test_true_for_declared_convention_with_levels(self) -> None:
        assert attributes_declare_multiscales(_multiscales_attrs()) is True

    def test_matches_convention_by_name_when_uuid_absent(self) -> None:
        attrs = _multiscales_attrs()
        attrs["zarr_conventions"] = [{"name": "multiscales"}]
        assert attributes_declare_multiscales(attrs) is True

    def test_false_for_flat_store(self) -> None:
        assert attributes_declare_multiscales({"proj:code": "EPSG:4326"}) is False

    def test_false_when_convention_declared_but_layout_empty(self) -> None:
        # Nothing extra for a client to read — advertising a pyramid would lie.
        attrs = _multiscales_attrs()
        attrs["multiscales"]["layout"] = []
        assert attributes_declare_multiscales(attrs) is False

    def test_false_when_layout_missing(self) -> None:
        attrs = _multiscales_attrs()
        del attrs["multiscales"]["layout"]
        assert attributes_declare_multiscales(attrs) is False

    def test_false_when_convention_missing_but_layout_present(self) -> None:
        attrs = _multiscales_attrs()
        attrs["zarr_conventions"] = []
        assert attributes_declare_multiscales(attrs) is False

    @pytest.mark.parametrize("conventions", ["multiscales", 42, None])
    def test_false_for_malformed_conventions(self, conventions: Any) -> None:
        assert attributes_declare_multiscales({"zarr_conventions": conventions}) is False


class TestReadRootAttributes:
    def test_reads_plain_zarr_v3_root(self, tmp_path: Path) -> None:
        store = _write_zarr_store(tmp_path / "store.zarr", _multiscales_attrs())
        assert read_root_attributes(str(store), icechunk=False)["multiscales"]["resampling_method"] == "mean"

    def test_empty_for_missing_store(self, tmp_path: Path) -> None:
        assert read_root_attributes(str(tmp_path / "absent.zarr"), icechunk=False) == {}

    def test_empty_for_unreadable_metadata(self, tmp_path: Path) -> None:
        store = tmp_path / "broken.zarr"
        store.mkdir()
        (store / "zarr.json").write_text("{not json", encoding="utf-8")
        assert read_root_attributes(str(store), icechunk=False) == {}

    def test_empty_for_remote_store(self) -> None:
        # Detection must not make a network round trip per STAC request.
        assert read_root_attributes("s3://bucket/store.zarr", icechunk=False) == {}


class TestZarrMediaType:
    def test_pyramided_store_advertises_profile(self, tmp_path: Path) -> None:
        store = _write_zarr_store(tmp_path / "pyramid.zarr", _multiscales_attrs())
        assert zarr_media_type(str(store), icechunk=False) == WEB_OPTIMIZED_ZARR_MEDIA_TYPE

    def test_flat_store_advertises_plain_type(self, tmp_path: Path) -> None:
        store = _write_zarr_store(tmp_path / "flat.zarr", {"proj:code": "EPSG:4326"})
        assert zarr_media_type(str(store), icechunk=False) == ZARR_V3_MEDIA_TYPE

    def test_falls_back_to_plain_type_when_store_absent(self, tmp_path: Path) -> None:
        assert zarr_media_type(str(tmp_path / "absent.zarr"), icechunk=False) == ZARR_V3_MEDIA_TYPE

    def test_falls_back_to_plain_type_when_detection_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A broken/locked store must never take the STAC collection down with it.
        def _boom(*_args: object, **_kwargs: object) -> dict[str, Any]:
            raise RuntimeError("store is locked")

        monkeypatch.setattr("open_geo_service.stac.media_types.read_root_attributes", _boom)
        assert zarr_media_type("/tmp/whatever.icechunk", icechunk=True) == ZARR_V3_MEDIA_TYPE


class TestIcechunkDetectionEndToEnd:
    """The Icechunk path against a store written by the real pipeline.

    The rest of this module builds root metadata by hand, which cannot catch a mismatch between
    what OGS *writes* and what this module *looks for* — the two are on opposite sides of the
    contract. These write real stores and read them back.
    """

    def _cube(self, ny: int, nx: int) -> Any:
        xr = pytest.importorskip("xarray")
        np = pytest.importorskip("numpy")
        pytest.importorskip("rioxarray")
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

        ds = xr.Dataset(
            {"v": (("t", "y", "x"), np.ones((1, ny, nx), dtype="float32"))},
            coords={
                "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
                "y": -1.0 - np.arange(ny) * 0.001,
                "x": 28.8 + np.arange(nx) * 0.001,
            },
        )
        return ds.rio.write_crs("EPSG:4326")

    def _write(self, tmp_path: Path, ny: int, nx: int) -> str:
        pytest.importorskip("icechunk")
        from open_geo_service.data_manager.services.downloader import write_to_icechunk_store

        store_path = tmp_path / f"{ny}x{nx}.icechunk"
        write_to_icechunk_store(self._cube(ny, nx), store_path, t_dim="t", crs="EPSG:4326")
        return str(store_path)

    def test_a_pyramided_store_written_by_ocs_is_detected(self, tmp_path: Path) -> None:
        # Past the 2048x2048 pyramid threshold, so the writer builds a multiscales layout.
        store = self._write(tmp_path, 2100, 2400)
        assert zarr_media_type(store, icechunk=True) == WEB_OPTIMIZED_ZARR_MEDIA_TYPE

    def test_a_flat_store_written_by_ocs_is_not(self, tmp_path: Path) -> None:
        store = self._write(tmp_path, 64, 64)
        assert zarr_media_type(store, icechunk=True) == ZARR_V3_MEDIA_TYPE

    def test_reads_the_root_group_not_pyramid_level_zero(self, tmp_path: Path) -> None:
        """The store accessors descend into level 0, whose attrs carry no multiscales layout.

        If detection ever read through those, every pyramided store would look flat — so pin
        that the layout is genuinely absent one level down.
        """
        zarr = pytest.importorskip("zarr")
        icechunk = pytest.importorskip("icechunk")

        store = self._write(tmp_path, 2100, 2400)
        repo = icechunk.Repository.open(icechunk.local_filesystem_storage(store))
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")

        assert attributes_declare_multiscales(dict(root.attrs)) is True
        assert attributes_declare_multiscales(dict(root["0"].attrs)) is False
