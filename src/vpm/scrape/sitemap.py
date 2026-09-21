"""Harvest Myntra product URLs from the public sitemap index.

Myntra product URLs carry their own metadata:

    https://www.myntra.com/<Category>/<Brand>/<slug>/<product_id>/buy

so the catalogue can be filtered down to footwear from the sitemaps alone,
without fetching a single product page. That is what makes a 100k-item
catalogue tractable: we only pay for PDP fetches on URLs we actually want.
"""

from __future__ import annotations

import gzip
import io
import re
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Iterator
from urllib.parse import unquote

import requests

SITEMAP_INDEX = "https://www.myntra.com/sitemap-index.xml.gz"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Categories that count as footwear. Taken from the live sitemap tally rather
# than guessed -- these are the article-type segments that actually appear.
FOOTWEAR_CATEGORIES = {
    "Casual-Shoes",
    "Sports-Shoes",
    "Sneakers",
    "Formal-Shoes",
    "Boots",
    "Sandals",
    "Sports-Sandals",
    "Flip-Flops",
    "Heels",
    "Flats",
}

_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
# .../<Category>/<Brand>/<slug>/<id>/buy
_PRODUCT = re.compile(
    r"^https://www\.myntra\.com/([^/]+)/([^/]+)/([^/]+)/(\d+)/buy/?$"
)


@dataclass(frozen=True)
class ProductRef:
    """A product URL decomposed into its parts. No page fetch required."""

    product_id: int
    category: str
    brand: str
    slug: str
    url: str

    def as_dict(self) -> dict:
        return asdict(self)


class Fetcher:
    """Rate-limited HTTP with retries.

    Politeness is not optional here: we are pulling tens of thousands of URLs
    from someone else's site. Default is ~5 req/s with backoff on failure.
    """

    def __init__(self, delay: float = 0.2, retries: int = 3, timeout: int = 60):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        self._last = 0.0

    def get(self, url: str) -> bytes | None:
        for attempt in range(self.retries):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=self.timeout)
                self._last = time.monotonic()
                if r.status_code == 200:
                    return r.content
                # 404 is final; anything else may be transient throttling.
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                self._last = time.monotonic()
            time.sleep(2**attempt)
        return None


def _gunzip(raw: bytes) -> str:
    """Sitemaps are gzipped, but tolerate a server that already decompressed."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return raw.decode("utf-8", errors="replace")


def list_sitemaps(fetcher: Fetcher, index_url: str = SITEMAP_INDEX) -> list[str]:
    raw = fetcher.get(index_url)
    if raw is None:
        raise RuntimeError(f"could not fetch sitemap index: {index_url}")
    return _LOC.findall(_gunzip(raw))


def parse_sitemap(text: str) -> Iterator[ProductRef]:
    """Yield a ProductRef for every product URL in one sitemap."""
    for loc in _LOC.findall(text):
        m = _PRODUCT.match(loc)
        if not m:
            continue  # editorial / category / landing page
        category, brand, slug, pid = m.groups()
        yield ProductRef(
            product_id=int(pid),
            category=category,
            brand=unquote(brand.replace("+", " ")).strip(),
            slug=slug,
            url=loc,
        )


def harvest(
    fetcher: Fetcher,
    sitemaps: Iterable[str],
    categories: set[str] | None = FOOTWEAR_CATEGORIES,
    limit: int | None = None,
) -> Iterator[ProductRef]:
    """Stream matching ProductRefs across sitemaps, de-duplicated by product_id.

    `categories=None` disables filtering (used for the Tier-2 distractor pool
    and for measuring category supply).
    """
    seen: set[int] = set()
    for sm_url in sitemaps:
        raw = fetcher.get(sm_url)
        if raw is None:
            continue
        for ref in parse_sitemap(_gunzip(raw)):
            if categories is not None and ref.category not in categories:
                continue
            if ref.product_id in seen:
                continue
            seen.add(ref.product_id)
            yield ref
            if limit is not None and len(seen) >= limit:
                return
