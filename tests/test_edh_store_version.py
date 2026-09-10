"""OCS reads only Zarr v3 Earth Data Hub stores (CLIM-955).

EDH leaves superseded stores readable and applies updates only to its v3 ones, so an older
store does not fail — it silently stops advancing, and an ingest against it reports no new
periods. That is indistinguishable from "nothing has been published yet", the same shape as
CLIM-952. Refusing is what makes it visible.

The three-way verdict is the crux of these tests: *not v3* and *cannot tell* must not be
confused, or a transient network fault becomes a refused ingest.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.datasets import era5_land

_STORE = "https://api.earthdatahub.destine.eu/era5/store.zarr"


@pytest.fixture(autouse=True)
def _clean_probe_state() -> Any:
    """Start each test with cold caches, and let `caplog` see the package logger.

    `startup.py` sets `propagate = False` on the `open_climate_service` logger so the package
    owns its handler and does not double-log under uvicorn. That also means caplog, which
    attaches at the root, sees nothing — the record is emitted and the assertion still fails.
    Re-enabled per test rather than weakened in the package.
    """
    package_logger = logging.getLogger("open_climate_service")
    previous = package_logger.propagate
    package_logger.propagate = True
    era5_land._edh_is_v3_cache.clear()
    era5_land._edh_unknown_until.clear()
    yield
    package_logger.propagate = previous
    era5_land._edh_is_v3_cache.clear()
    era5_land._edh_unknown_until.clear()


def _dataset(last: str = "2026-05-31T23:00") -> xr.Dataset:
    times = np.array(["2026-05-31T22:00", last], dtype="datetime64[ns]")
    return xr.Dataset({"t2m": (("valid_time",), [1.0, 2.0])}, coords={"valid_time": times})


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _serving(payload: object) -> Any:
    def fake_urlopen(request: Any, timeout: int = 0) -> _Response:
        return _Response(payload)

    return fake_urlopen


def _raising(error: BaseException) -> Any:
    def fake_urlopen(request: Any, timeout: int = 0) -> _Response:
        raise error

    return fake_urlopen


def _http_error(code: int) -> HTTPError:
    return HTTPError(_STORE, code, "nope", {}, None)  # type: ignore[arg-type]


# --- the verdict ------------------------------------------------------------------------


def test_a_v3_store_is_recognised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(era5_land, "urlopen", _serving({"zarr_format": 3}))

    assert era5_land._edh_is_zarr_v3(_STORE) is True


def test_a_missing_metadata_document_means_not_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 on `zarr.json` is an answer: there is no v3 metadata here."""
    monkeypatch.setattr(era5_land, "urlopen", _raising(_http_error(404)))

    assert era5_land._edh_is_zarr_v3(_STORE) is False


def test_an_older_declared_format_means_not_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(era5_land, "urlopen", _serving({"zarr_format": 2}))

    assert era5_land._edh_is_zarr_v3(_STORE) is False


@pytest.mark.parametrize("error", [_http_error(500), _http_error(429), _http_error(401), OSError("timeout")])
def test_an_unreachable_store_is_unknown_not_non_v3(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """The distinction that matters: a network fault is not evidence about the format.

    Collapsing it into "not v3" would refuse a store that reads perfectly well, every time
    the network hiccups.
    """
    monkeypatch.setattr(era5_land, "urlopen", _raising(error))

    assert era5_land._edh_is_zarr_v3(_STORE) is None


@pytest.mark.parametrize("payload", [[], {"metadata": []}, "a string", None, {"zarr_format": "three"}])
def test_a_response_that_declares_nothing_is_unknown(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    """A 200 of an unexpected shape — a proxy error page, say — says nothing about the store."""
    monkeypatch.setattr(era5_land, "urlopen", _serving(payload))

    assert era5_land._edh_is_zarr_v3(_STORE) is None


# --- refusing ---------------------------------------------------------------------------


class _ClosableDataset:
    """Stands in for an opened store so the close can be observed.

    `xr.Dataset` uses `__slots__`, so `close` cannot be monkeypatched onto a real one. This
    exposes only what `_require_zarr_v3` touches.
    """

    def __init__(self, last: str = "2026-05-31T23:00") -> None:
        self._inner = _dataset(last)
        self.closed = False

    @property
    def coords(self) -> Any:
        return self._inner.coords

    @property
    def sizes(self) -> Any:
        return self._inner.sizes

    def __getitem__(self, key: str) -> Any:
        return self._inner[key]

    def close(self) -> None:
        self.closed = True


def test_a_non_v3_store_is_refused_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused store must not be left open — the handle would leak on every period."""
    monkeypatch.setattr(era5_land, "_edh_is_zarr_v3", lambda _url: False)
    ds = _ClosableDataset()

    with pytest.raises(RuntimeError) as raised:
        era5_land._require_zarr_v3(_STORE, ds)  # type: ignore[arg-type]

    assert "not Zarr v3" in str(raised.value)
    assert "2026-05-31T23:00" in str(raised.value), "the operator needs the gap, not just the verdict"
    assert "CLIM-955" in str(raised.value)
    assert ds.closed, "the refused store was left open"


def test_a_v3_store_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(era5_land, "_edh_is_zarr_v3", lambda _url: True)

    era5_land._require_zarr_v3(_STORE, _dataset())


def test_an_undeterminable_store_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown must never become a refusal — that is a working store broken by a hiccup."""
    monkeypatch.setattr(era5_land, "_edh_is_zarr_v3", lambda _url: None)

    era5_land._require_zarr_v3(_STORE, _dataset())


def test_a_failing_timestamp_read_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enrichment is optional; the refusal is not.

    Reading the latest timestamp is a lazy remote fetch that can fail on its own. Losing it
    must cost the timestamp and nothing else.
    """
    monkeypatch.setattr(era5_land, "_edh_is_zarr_v3", lambda _url: False)

    class _Exploding:
        def isel(self, *args: object, **kwargs: object) -> Any:
            raise OSError("coordinate chunk could not be fetched")

    ds = _dataset()
    monkeypatch.setitem(ds._variables, "valid_time", _Exploding())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError) as raised:
        era5_land._require_zarr_v3(_STORE, ds)

    assert "not Zarr v3" in str(raised.value)
    assert "latest timestamp" not in str(raised.value)


def test_a_store_with_no_time_axis_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(era5_land, "_edh_is_zarr_v3", lambda _url: False)

    with pytest.raises(RuntimeError, match="not Zarr v3"):
        era5_land._require_zarr_v3(_STORE, xr.Dataset({"t2m": 1.0}))


# --- probe cost -------------------------------------------------------------------------


def test_an_unknown_verdict_is_not_reprobed_on_every_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_latest_available` reopens the store for every period.

    Without a negative cache, a store fsspec can read while this probe times out costs 15 s
    *per period* — the check becomes the bottleneck during exactly the transient failure it
    is meant to tolerate.
    """
    calls = {"n": 0}

    def counting(request: Any, timeout: int = 0) -> _Response:
        calls["n"] += 1
        raise OSError("timeout")

    monkeypatch.setattr(era5_land, "urlopen", counting)

    for _ in range(100):
        assert era5_land._edh_is_zarr_v3(_STORE) is None

    assert calls["n"] == 1, f"probed {calls['n']} times across 100 opens"


def test_the_negative_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bounded, not permanent: the check must return without a process restart."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(era5_land, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(era5_land, "urlopen", _raising(OSError("down")))
    assert era5_land._edh_is_zarr_v3(_STORE) is None

    clock["t"] += era5_land._EDH_UNKNOWN_TTL_SECONDS + 1
    monkeypatch.setattr(era5_land, "urlopen", _serving({"zarr_format": 3}))

    assert era5_land._edh_is_zarr_v3(_STORE) is True, "the negative cache never expired"


def test_a_confirmed_verdict_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def counting(request: Any, timeout: int = 0) -> _Response:
        calls["n"] += 1
        return _Response({"zarr_format": 3})

    monkeypatch.setattr(era5_land, "urlopen", counting)
    for _ in range(10):
        assert era5_land._edh_is_zarr_v3(_STORE) is True

    assert calls["n"] == 1


# --- authentication ---------------------------------------------------------------------


def test_the_env_key_path_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`urlopen` does not turn `user:pass@host` userinfo into Basic Auth the way fsspec does.

    Probing the authenticated URL therefore 401s for anyone using EDH_API_KEY — the whole
    documented environment-variable path — and the check silently never runs.
    """
    import base64

    monkeypatch.setenv("EDH_API_KEY", "mytoken")

    header = era5_land._edh_auth_header(_STORE)

    assert base64.b64decode(header["Authorization"].split()[1]).decode() == "edh:mytoken"


def test_the_netrc_path_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`urlopen` does not read netrc either, though fsspec does via `trust_env=True`."""
    import base64

    monkeypatch.delenv("EDH_API_KEY", raising=False)

    class _Netrc:
        @staticmethod
        def authenticators(host: str) -> tuple[str, str, str] | None:
            # The shape a real netrc yields for the entry EDH documents: a password and no
            # login. The previous fixture returned ("edh", "", "secret"), which netrc never
            # produces for that entry, and it hid the empty-login case entirely.
            return ("", "", "secret") if host == "api.earthdatahub.destine.eu" else None

    monkeypatch.setattr(era5_land.netrc, "netrc", lambda: _Netrc())

    header = era5_land._edh_auth_header(_STORE)

    assert base64.b64decode(header["Authorization"].split()[1]).decode() == "edh:secret"


def test_no_credentials_anywhere_sends_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDH_API_KEY", raising=False)
    monkeypatch.setattr(era5_land.netrc, "netrc", lambda: (_ for _ in ()).throw(OSError("no netrc")))

    assert era5_land._edh_auth_header(_STORE) == {}


def test_the_probe_never_puts_credentials_in_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Userinfo in the probe URL is both useless to urlopen and a leak into anything logging."""
    monkeypatch.setenv("EDH_API_KEY", "mytoken")
    seen: list[str] = []

    def capturing(request: Any, timeout: int = 0) -> _Response:
        seen.append(request.full_url)
        return _Response({"zarr_format": 3})

    monkeypatch.setattr(era5_land, "urlopen", capturing)
    era5_land._edh_is_zarr_v3(_STORE)

    assert seen and all("mytoken" not in url and "@" not in url for url in seen)


def test_an_explicit_netrc_login_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaulting an empty login must not override one that was actually given."""
    import base64

    monkeypatch.delenv("EDH_API_KEY", raising=False)

    class _Netrc:
        @staticmethod
        def authenticators(host: str) -> tuple[str, str, str] | None:
            return ("someone", "", "secret")

    monkeypatch.setattr(era5_land.netrc, "netrc", lambda: _Netrc())

    header = era5_land._edh_auth_header(_STORE)

    assert base64.b64decode(header["Authorization"].split()[1]).decode() == "someone:secret"
