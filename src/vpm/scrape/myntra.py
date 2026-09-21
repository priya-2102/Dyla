"""Parse a Myntra product page into a catalogue record.

Everything we need is in the `window.__myx` JSON blob the page ships with, so
one GET per product yields metadata *and* the full multi-view image list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

# window.__myx = {...};  followed by the closing script tag.
_MYX = re.compile(r"window\.__myx\s*=\s*(\{.*?\})\s*;?\s*</script>", re.S)

# Image URLs are templates: .../h_($height),q_($qualityPercentage),w_($width)/v1/...
_TPL = re.compile(r"\(\$(height|width|qualityPercentage)\)")


@dataclass
class CatalogueItem:
    product_id: int
    name: str
    brand: str
    base_colour: str | None
    article_type: str | None
    sub_category: str | None
    master_category: str | None
    gender: str | None
    images: list[str] = field(default_factory=list)
    # Hard-negative graph: same model in other colourways, and related styles.
    colour_variants: list[int] = field(default_factory=list)
    related_styles: list[int] = field(default_factory=list)
    source_url: str = ""
    fetched_at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_image_url(src: str, width: int = 1080, height: int = 1440, quality: int = 90) -> str:
    """Fill a Myntra image template and force https.

    The template lets us fetch catalogue images at exactly the resolution the
    backbone wants, instead of downloading full-size and resizing locally.
    """
    subs = {"height": str(height), "width": str(width), "qualityPercentage": str(quality)}
    url = _TPL.sub(lambda m: subs[m.group(1)], src)
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def _ids(blob) -> list[int]:
    """Pull integer product ids out of the loosely-typed related-item lists.

    Myntra is inconsistent here -- entries may be bare ids, or dicts keyed by
    any of several id fields -- so accept whatever shape shows up.
    """
    out: list[int] = []
    if not isinstance(blob, list):
        return out
    for entry in blob:
        if isinstance(entry, int):
            out.append(entry)
        elif isinstance(entry, dict):
            for key in ("styleid", "styleId", "id", "productId"):
                val = entry.get(key)
                if isinstance(val, int):
                    out.append(val)
                    break
                if isinstance(val, str) and val.isdigit():
                    out.append(int(val))
                    break
    return out


def extract_myx(html: str) -> dict | None:
    m = _MYX.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_product(
    html: str,
    source_url: str = "",
    fetched_at: str = "",
    width: int = 1080,
    height: int = 1440,
    quality: int = 90,
) -> CatalogueItem | None:
    myx = extract_myx(html)
    if not myx:
        return None
    pdp = myx.get("pdpData") or {}
    pid = pdp.get("id")
    if not isinstance(pid, int):
        return None

    analytics = pdp.get("analytics") or {}
    brand = (pdp.get("brand") or {}).get("name") or analytics.get("brand") or ""

    images: list[str] = []
    for album in (pdp.get("media") or {}).get("albums") or []:
        # The 'default' album holds the product shots; 'animatedImage' and
        # friends are marketing assets we do not want in the index.
        if album.get("name") and album["name"] != "default":
            continue
        for img in album.get("images") or []:
            src = img.get("src")
            if src:
                images.append(resolve_image_url(src, width, height, quality))

    # De-duplicate while preserving view order.
    images = list(dict.fromkeys(images))

    return CatalogueItem(
        product_id=pid,
        name=pdp.get("name") or "",
        brand=brand,
        base_colour=pdp.get("baseColour"),
        article_type=analytics.get("articleType"),
        sub_category=analytics.get("subCategory"),
        master_category=analytics.get("masterCategory"),
        gender=analytics.get("gender"),
        images=images,
        colour_variants=_ids(pdp.get("colours")),
        related_styles=_ids(pdp.get("relatedStyles")),
        source_url=source_url,
        fetched_at=fetched_at,
    )
