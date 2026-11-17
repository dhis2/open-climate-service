"""Early-boot side effects.

This module is imported before any other open_geo_service modules so that
environment variables and logging are configured before other imports.
"""

import logging
import os

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# -- open_geo_service / third-party logging setup ---------------------------------------
eo_logger = logging.getLogger("open_geo_service")
eo_logger.setLevel(logging.INFO)
if not eo_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    eo_logger.addHandler(handler)
eo_logger.propagate = False


def _ensure_proj_database() -> None:
    """Make sure pyproj can resolve CRS codes, healing a polluted host PROJ env.

    Common failure mode on developer machines: a conda/miniforge ``PROJ_DATA``
    (or ``PROJ_LIB``) is inherited that points at a ``proj.db`` incompatible with
    the installed pyproj, so even ``EPSG:4326`` fails to resolve — which then
    breaks GeoZarr metadata generation during ingestion with a confusing
    "proj:code 'EPSG:4326' does not resolve to a known CRS" error.

    When CRS resolution already works (e.g. a clean Docker image) this is a no-op.
    Otherwise it pins the PROJ database to the one bundled with pyproj.
    """
    try:
        import pyproj
        from pyproj import CRS, datadir
    except ImportError:
        return  # client-only install without pyproj

    try:
        CRS.from_authority("EPSG", "4326")
        return  # resolution already works — leave the environment untouched
    except Exception:
        pass

    bundled = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
    if not os.path.exists(os.path.join(bundled, "proj.db")):
        eo_logger.error(
            "PROJ database self-check failed (EPSG:4326 does not resolve) and no bundled "
            "pyproj database was found at %s. CRS operations (ingest/STAC) will fail; "
            "check PROJ_DATA / PROJ_LIB.",
            bundled,
        )
        return

    os.environ["PROJ_DATA"] = bundled
    os.environ.pop("PROJ_LIB", None)
    try:
        datadir.set_data_dir(bundled)
        CRS.from_authority("EPSG", "4326")
    except Exception:
        eo_logger.exception("Could not establish a working PROJ database even from bundled pyproj data")
        return
    eo_logger.warning(
        "Inherited PROJ environment could not resolve EPSG:4326; pinned the PROJ "
        "database to pyproj's bundled data at %s",
        bundled,
    )


_ensure_proj_database()
