"""Build the catalogue: sitemap refs -> product records -> local images.

Two tiers, mirroring the Oxford5k/Oxford105k distractor convention:

  Tier 1  core      multi-view, full metadata -- all accuracy evaluation
  Tier 2  scale     single view, minimal metadata -- latency/scale work and the
                    distractor database the refusal scorer normalises against

Every stage is resumable and append-only, so a run can be interrupted and
restarted without losing or duplicating work.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import requests

from ..scrape.download import download_images, image_path
from ..scrape.myntra import CatalogueItem, parse_product
from ..scrape.sitemap import UA, Fetcher, ProductRef, harvest, list_sitemaps


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def harvest_refs(
    out: Path,
    n_sitemaps: int | None = None,
    categories: set[str] | None = None,
    limit: int | None = None,
    delay: float = 0.2,
) -> int:
    """Stage 1: collect product URLs, filtered to footwear, no PDP fetches."""
    fetcher = Fetcher(delay=delay)
    sitemaps = list_sitemaps(fetcher)
    if n_sitemaps is not None:
        sitemaps = sitemaps[:n_sitemaps]
    refs = harvest(fetcher, sitemaps, categories=categories, limit=limit)
    return write_jsonl(out, (r.as_dict() for r in refs))


class _RateLimiter:
    """Shared minimum-interval gate across worker threads."""

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.delay
        if sleep_for:
            time.sleep(sleep_for)


def _fetch_pdp(
    session: requests.Session,
    limiter: _RateLimiter,
    ref: dict,
    width: int,
    height: int,
    quality: int,
    timeout: int = 45,
) -> CatalogueItem | None:
    for attempt in range(3):
        limiter.wait()
        try:
            r = session.get(ref["url"], timeout=timeout)
            if r.status_code == 200:
                return parse_product(
                    r.text,
                    source_url=ref["url"],
                    fetched_at=_now(),
                    width=width,
                    height=height,
                    quality=quality,
                )
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    return None


def fetch_items(
    refs_path: Path,
    out: Path,
    workers: int = 6,
    delay: float = 0.12,
    width: int = 1080,
    height: int = 1440,
    quality: int = 90,
    limit: int | None = None,
    progress_every: int = 250,
) -> int:
    """Stage 2: fetch product pages and write catalogue records.

    Resumable -- product ids already present in `out` are skipped, and new rows
    are appended, so re-running after an interruption picks up where it stopped.
    """
    done = {row["product_id"] for row in read_jsonl(out)}
    todo = [r for r in read_jsonl(refs_path) if r["product_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    limiter = _RateLimiter(delay)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    lock = threading.Lock()
    with out.open("a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(_fetch_pdp, session, limiter, ref, width, height, quality)
            for ref in todo
        ]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            item = fut.result()
            if item is None or not item.images:
                continue
            with lock:
                fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
                written += 1
                if progress_every and written % progress_every == 0:
                    fh.flush()
                    print(f"  {written} items written ({i}/{len(todo)} fetched)", flush=True)
    return written


def download_catalogue_images(
    items_path: Path,
    image_root: Path,
    max_views: int | None = None,
    workers: int = 8,
) -> dict:
    """Stage 3: pull the images for every catalogue record."""
    jobs: list[tuple[str, Path]] = []
    for row in read_jsonl(items_path):
        urls = row["images"]
        if max_views is not None:
            urls = urls[:max_views]
        for view, url in enumerate(urls):
            jobs.append((url, image_path(image_root, row["product_id"], view)))
    res = download_images(jobs, workers=workers)
    return {"jobs": len(jobs), "ok": res.ok, "skipped": res.skipped, "failed": res.failed}
