"""The credentials each provider builds from its environment variables."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from open_climate_service.plugins.rasters.providers import cds, cdse, edh

_KEYS = ("EDH_API_KEY", "CDS_API_KEY", "CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY")


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEYS:
        monkeypatch.setenv(name, name.lower())


def _cdse_options() -> dict[str, str]:
    return cdse.rasterio_session().get_credential_options()


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        # base64 of "edh:edh_api_key"
        (edh.auth_headers, {"Authorization": "Basic ZWRoOmVkaF9hcGlfa2V5"}),
        (cds.auth_headers, {"Authorization": "Bearer cds_api_key"}),
        (_cdse_options, {"AWS_ACCESS_KEY_ID": "cdse_s3_access_key", "AWS_SECRET_ACCESS_KEY": "cdse_s3_secret_key"}),
    ],
)
def test_each_provider_builds_its_credentials_from_the_environment(
    build: Callable[[], dict[str, str]], expected: dict[str, str]
) -> None:
    assert build() == expected


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (edh.auth_headers, "EDH_API_KEY"),
        (cds.auth_headers, "CDS_API_KEY"),
        (_cdse_options, "CDSE_S3_ACCESS_KEY"),
        (_cdse_options, "CDSE_S3_SECRET_KEY"),
    ],
)
def test_a_missing_key_is_named(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[], dict[str, str]], name: str
) -> None:
    monkeypatch.setenv(name, "  ")

    with pytest.raises(PermissionError, match=f"{name} is not set"):
        build()
