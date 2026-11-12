"""Throughput and latency of `/zarr` chunk reads against a Nepal-sized store (CLIM-1580).

Not a test: run it by hand before and after a change to the chunk path, and record the output
in the PR.

    uv run python tests/benchmarks/zarr_chunk_reads.py [--concurrency 6] [--rounds 5]

It writes a daily CHIRPS-shaped store over Nepal's extent (3 years at 0.05 degrees) with OCS's
own writer into a temporary data root, registers it as a published dataset, serves the app with
uvicorn and requests every chunk of the store, `--rounds` times, `--concurrency` at a time.
Each round is one map view's worth of chunk requests, repeated. A second pass asks for the first
kilobyte of each chunk, the way a client reads a shard index.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import socket
import statistics
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import uvicorn
import xarray as xr

from open_climate_service.data_manager.services.downloader import write_to_icechunk_store
from open_climate_service.ingestions import services
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.main import app

DATASET_ID = "chirps3_precipitation_daily"


def _write_store(root: Path) -> tuple[Path, list[str]]:
    times = pd.date_range("2023-01-01", periods=1096, freq="D")
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), np.random.default_rng(0).random((len(times), 84, 165), dtype="float32"))},
        coords={"t": times, "y": np.linspace(30.5, 26.3, 84), "x": np.linspace(80.0, 88.25, 165)},
    )
    store_path = root / "rasters" / f"{DATASET_ID}.icechunk"
    write_to_icechunk_store(ds, store_path, crs="EPSG:4326")

    import icechunk
    import zarr

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    array = zarr.open_array(repo.readonly_session("main").store, path="precip", mode="r")
    grid = [-(-size // chunk) for size, chunk in zip(array.shape, array.chunks, strict=True)]
    keys = [f"precip/c/{t}/{y}/{x}" for t in range(grid[0]) for y in range(grid[1]) for x in range(grid[2])]
    return store_path, keys


def _register(store_path: Path, artifacts_dir: Path) -> None:
    services.ARTIFACTS_DIR = artifacts_dir
    services.ARTIFACTS_INDEX_PATH = artifacts_dir / "records.json"
    services._save_records(
        [
            ArtifactRecord(
                artifact_id="bench",
                dataset_id=DATASET_ID,
                dataset_name="CHIRPS3 precipitation",
                variable="precip",
                format=ArtifactFormat.ICECHUNK,
                path=str(store_path),
                asset_paths=[str(store_path)],
                variables=["precip"],
                request_scope=ArtifactRequestScope(
                    start="2023-01-01", end="2025-12-31", bbox=(80.0, 26.3, 88.25, 30.5)
                ),
                coverage=ArtifactCoverage(
                    temporal=CoverageTemporal(start="2023-01-01", end="2025-12-31"),
                    spatial=CoverageSpatial(xmin=80.0, ymin=26.3, xmax=88.25, ymax=30.5),
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=DATASET_ID),
            )
        ]
    )
    # Measured now, so the app's startup pass has nothing to write while chunks are being read.
    services.measure_missing_statistics(lambda: False)


def _serve() -> tuple[str, uvicorn.Server, threading.Thread]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{sock.getsockname()[1]}", server, thread


async def _run(base: str, keys: list[str], concurrency: int, rounds: int, ranged: bool) -> list[float]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    headers = {"Range": "bytes=0-1023"} if ranged else {}
    latencies: list[float] = []
    gate = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=60) as client:

        async def fetch(key: str) -> None:
            async with gate:
                started = time.perf_counter()
                response = await client.get(f"/zarr/{DATASET_ID}/{key}", headers=headers)
                latencies.append(time.perf_counter() - started)
                assert response.status_code == (206 if ranged else 200), (key, response.status_code)

        # The map's metadata request comes first and warms the record lookup.
        assert (await client.get(f"/zarr/{DATASET_ID}/zarr.json")).status_code == 200
        for _ in range(rounds):
            await asyncio.gather(*(fetch(key) for key in keys))
    return latencies


def _client(base: str, keys: list[str], concurrency: int, rounds: int, ranged: bool) -> tuple[list[float], float]:
    """Run the requests in their own process, so the client does not share the server's GIL."""
    started = time.perf_counter()
    latencies = asyncio.run(_run(base, keys, concurrency, rounds, ranged))
    return latencies, time.perf_counter() - started


def _report(label: str, latencies: list[float], seconds: float) -> None:
    ordered = sorted(latencies)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    print(
        f"{label:<14} {len(latencies):>5} requests  {len(latencies) / seconds:>7.0f} req/s  "
        f"p50 {statistics.median(ordered) * 1000:>6.1f} ms  p95 {p95 * 1000:>6.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput and latency of /zarr chunk reads (CLIM-1580).")
    parser.add_argument("--concurrency", type=int, default=6, help="parallel requests per client, like a browser")
    parser.add_argument("--clients", type=int, default=1, help="client processes; one saturates above ~8 requests")
    parser.add_argument("--rounds", type=int, default=5, help="times each client requests every chunk")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store_path, keys = _write_store(root)
        _register(store_path, root / "artifacts")
        base, server, thread = _serve()
        try:
            print(f"{len(keys)} chunks, {args.clients} client(s) x {args.concurrency} parallel, {args.rounds} rounds")
            with ProcessPoolExecutor(args.clients, mp_context=multiprocessing.get_context("spawn")) as clients:
                for label, ranged in (("full chunks", False), ("first 1 KiB", True)):
                    runs = [
                        clients.submit(_client, base, keys, args.concurrency, args.rounds, ranged).result
                        for _ in range(args.clients)
                    ]
                    results = [run() for run in runs]
                    # Timed inside each client, so starting the client processes is not counted.
                    _report(label, [lat for lats, _ in results for lat in lats], max(secs for _, secs in results))
        finally:
            server.should_exit = True
            thread.join(timeout=10)


if __name__ == "__main__":
    main()
