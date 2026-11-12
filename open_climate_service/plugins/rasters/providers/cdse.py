"""Copernicus Data Space Ecosystem (CDSE) S3 credentials."""

from typing import Any

from rasterio.session import Session

from open_climate_service.plugins.rasters.providers import require_env


def api_key() -> dict[str, str]:
    """The CDSE S3 access and secret keys."""
    hint = (
        "Create free CDSE S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu; "
        "the secret key is shown only once."
    )
    return {name: require_env(name, hint=hint) for name in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY")}


class _KeySession(Session):
    """Hands GDAL static S3 keys.

    rasterio refuses AWS keys as plain ``Env`` options, and its ``AWSSession`` needs boto3, which
    OCS does not install.
    """

    def __init__(self, credentials: dict[str, str]) -> None:
        self.credentials = credentials

    # The base methods are unannotated and return NotImplemented, which pyright takes as their type.
    @classmethod
    def hascreds(cls, config: dict[str, Any]) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return "AWS_ACCESS_KEY_ID" in config

    def get_credential_options(self) -> dict[str, str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return dict(self.credentials)


def rasterio_session() -> Session:
    """A session for ``rasterio.Env(session=...)`` that reads the CDSE ``eodata`` bucket."""
    keys = api_key()
    return _KeySession(
        {"AWS_ACCESS_KEY_ID": keys["CDSE_S3_ACCESS_KEY"], "AWS_SECRET_ACCESS_KEY": keys["CDSE_S3_SECRET_KEY"]}
    )
