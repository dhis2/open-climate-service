"""Tests for the PROJ database self-heal in open_geo_service.startup."""

from __future__ import annotations

import os

import pytest

pyproj = pytest.importorskip("pyproj")


def _bundled_proj_dir() -> str:
    return os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")


def test_ensure_proj_database_pins_bundled_db_when_resolution_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the inherited PROJ env can't resolve EPSG:4326, pin the bundled pyproj db."""
    from pyproj import CRS

    from open_geo_service import startup

    bundled = _bundled_proj_dir()
    if not os.path.exists(os.path.join(bundled, "proj.db")):
        pytest.skip("pyproj bundled proj.db not available in this environment")

    real_from_authority = CRS.from_authority
    state = {"first_call": True}

    def flaky_from_authority(auth: str, code: str, *args: object, **kwargs: object):
        # Simulate a broken inherited PROJ database on the first check, then recover
        # once the bundled database has been pinned.
        if state["first_call"]:
            state["first_call"] = False
            raise pyproj.exceptions.CRSError("simulated broken PROJ database")
        return real_from_authority(auth, code)

    captured: dict[str, str] = {}
    monkeypatch.setattr(pyproj.CRS, "from_authority", flaky_from_authority)
    monkeypatch.setattr(pyproj.datadir, "set_data_dir", lambda d: captured.update(dir=d))
    monkeypatch.setenv("PROJ_DATA", "/some/broken/conda/path")  # restored on teardown

    startup._ensure_proj_database()

    assert captured["dir"] == bundled  # pinned pyproj's bundled database
    assert os.environ["PROJ_DATA"] == bundled  # and exported it for GDAL et al.


def test_ensure_proj_database_is_noop_when_resolution_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CRS resolution already works the env is left untouched."""
    from pyproj import CRS

    from open_geo_service import startup

    touched: dict[str, str] = {}
    monkeypatch.setattr(pyproj.datadir, "set_data_dir", lambda d: touched.update(dir=d))

    startup._ensure_proj_database()

    assert "dir" not in touched  # no fallback performed
    assert CRS.from_authority("EPSG", "4326").to_authority() == ("EPSG", "4326")
