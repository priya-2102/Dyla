"""Concurrent, resumable image download.

CDN image fetches tolerate more concurrency than product-page fetches, so this
runs its own pool rather than sharing the PDP fetcher's rate limit. Everything
is resumable: an existing non-empty file is never re-fetched, so an interrupted
run costs nothing to restart.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path

import requests

from .sitemap import UA


@dataclass
class DownloadResult:
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[str] = None

    def __post_init__(self):
        if self.failures is None:
            self.failures = []


def image_path(root: Path, product_id: int, view: int) -> Path:
    """Shard by the id's last two digits so no directory holds 100k entries."""
    return root / f"{product_id % 100:02d}" / str(product_id) / f"{view}.jpg"


def _fetch_one(session: requests.Session, url: str, dest: Path, timeout: int) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return "failed"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp file so an interrupted run never leaves a truncated
        # JPEG that the resume logic would mistake for a completed download.
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return "ok"
    except requests.RequestException:
        return "failed"


def download_images(
    jobs: list[tuple[str, Path]],
    workers: int = 8,
    timeout: int = 30,
) -> DownloadResult:
    """Fetch (url, destination) pairs. Returns counts and the failed URLs."""
    result = DownloadResult()
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    # A pool per session is fine; requests.Session is thread-safe for plain GETs
    # against the same host with a sized connection pool.
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, session, u, d, timeout): u for u, d in jobs}
        for fut in cf.as_completed(futures):
            status = fut.result()
            if status == "ok":
                result.ok += 1
            elif status == "skipped":
                result.skipped += 1
            else:
                result.failed += 1
                result.failures.append(futures[fut])
    return result
