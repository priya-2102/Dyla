"""Encode the catalogue into a searchable embedding matrix.

Cost is asymmetric and that asymmetry is the whole strategy: catalogue encoding
happens once, offline, on MPS in large batches (~7-23 ms/image), while a query
is a single online forward pass. So anything that can be moved to the catalogue
side -- trimming, multi-view, mirror augmentation -- should be.
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from ..catalogue.trim import trim_to_square
from ..scrape.download import image_path
from .backbone import Backbone

# Some CDN JPEGs are truncated; better to use a partial image than to drop the view.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class CatalogueEmbeddings:
    emb: np.ndarray           # (N, D) float32, L2-normalised
    item_ids: np.ndarray      # (N,) product ids
    view_ids: np.ndarray      # (N,) which view of that item
    mirrored: np.ndarray      # (N,) bool: is this the mirrored copy
    backbone: str
    image_size: int
    pooling: str
    whiten_dim: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            emb=self.emb,
            item_ids=self.item_ids,
            view_ids=self.view_ids,
            mirrored=self.mirrored,
            meta=json.dumps(
                {
                    "backbone": self.backbone,
                    "image_size": self.image_size,
                    "pooling": self.pooling,
                    "whiten_dim": self.whiten_dim,
                }
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "CatalogueEmbeddings":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            emb=z["emb"],
            item_ids=z["item_ids"],
            view_ids=z["view_ids"],
            mirrored=z["mirrored"],
            backbone=meta["backbone"],
            image_size=meta["image_size"],
            pooling=meta["pooling"],
            whiten_dim=meta.get("whiten_dim", 0),
        )


def load_view(
    image_root: Path, product_id: int, view: int, size: int, trim: bool
) -> tuple[Image.Image, bool] | None:
    p = image_path(image_root, product_id, view)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        img = Image.open(p)
        img.load()
    except Exception:
        return None
    if trim:
        return trim_to_square(img, size=size)
    return img.convert("RGB").resize((size, size), Image.BICUBIC), False


def encode_catalogue(
    items_path: Path,
    image_root: Path,
    backbone: Backbone,
    max_views: int | None = None,
    trim: bool = True,
    mirror: bool = False,
    batch_size: int = 32,
    limit: int | None = None,
    progress_every: int = 2000,
    exclude: set[int] | None = None,
    workers: int = 8,
) -> CatalogueEmbeddings:
    """Encode every downloaded view. Missing images are skipped, not fatal.

    `mirror=True` also indexes a horizontally flipped copy of each view.
    Catalogue shots are near-always one shoe in a single orientation, so roughly
    half of real query photos show the opposite foot; mirroring the catalogue is
    cheaper than paying flip TTA on every query.
    """
    rows: list[tuple[int, int]] = []
    with items_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if exclude and row["product_id"] in exclude:
                continue
            n = len(row["images"]) if max_views is None else min(max_views, len(row["images"]))
            for v in range(n):
                rows.append((row["product_id"], v))
            if limit is not None and len({p for p, _ in rows}) >= limit:
                break

    embs: list[np.ndarray] = []
    meta_item: list[int] = []
    meta_view: list[int] = []
    meta_mirror: list[bool] = []
    n_trimmed = 0
    n_missing = 0

    buf_imgs: list[Image.Image] = []
    buf_meta: list[tuple[int, int, bool]] = []

    # Decoding a 1080x1440 JPEG and trimming it costs more wall-clock than the
    # forward pass does, and it is single-threaded. Measured before this change:
    # 5.5 views/s with the GPU ~90% idle. Pillow releases the GIL during decode,
    # so a thread pool prefetching images keeps the accelerator fed.
    def _prefetch(job):
        pid, view = job
        return pid, view, load_view(image_root, pid, view, backbone.cfg.image_size, trim)

    def flush() -> None:
        if not buf_imgs:
            return
        batch = torch.stack([backbone.transform(im) for im in buf_imgs])
        vecs = backbone.encode_tensor(batch).numpy()
        embs.append(vecs)
        for pid, vid, mir in buf_meta:
            meta_item.append(pid)
            meta_view.append(vid)
            meta_mirror.append(mir)
        buf_imgs.clear()
        buf_meta.clear()

    def _bounded_prefetch(jobs, workers, depth):
        """Sliding-window prefetch.

        `ThreadPoolExecutor.map` submits every task immediately, which for ~48k
        catalogue views means every decoded image is held in memory at once --
        tens of gigabytes, and the machine thrashes instead of encoding. Keep at
        most `depth` decodes in flight.
        """
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            it = iter(jobs)
            pending: deque = deque()
            for job in itertools.islice(it, depth):
                pending.append(pool.submit(_prefetch, job))
            while pending:
                fut = pending.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(pool.submit(_prefetch, nxt))
                yield fut.result()

    for i, (pid, view, got) in enumerate(
        _bounded_prefetch(rows, workers, depth=max(batch_size * 4, 64)), 1
    ):
        if got is None:
            n_missing += 1
            continue
        img, was_trimmed = got
        n_trimmed += int(was_trimmed)
        buf_imgs.append(img)
        buf_meta.append((pid, view, False))
        if mirror:
            buf_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            buf_meta.append((pid, view, True))
        if len(buf_imgs) >= batch_size:
            flush()
        if progress_every and i % progress_every == 0:
            print(f"  encoded {i}/{len(rows)} views (missing {n_missing})", flush=True)
    flush()

    print(f"  trimmed {n_trimmed} views; {n_missing} images missing/unreadable")
    return CatalogueEmbeddings(
        emb=np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, backbone.dim), np.float32),
        item_ids=np.array(meta_item, dtype=np.int64),
        view_ids=np.array(meta_view, dtype=np.int32),
        mirrored=np.array(meta_mirror, dtype=bool),
        backbone=backbone.name,
        image_size=backbone.cfg.image_size,
        pooling=backbone.cfg.pooling,
    )
