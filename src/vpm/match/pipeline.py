"""Single-query matching: photo in, ranked top-5 + confidence out.

Latency is dominated by the backbone forward pass -- exact search over the whole
catalogue is well under a millisecond -- so the cost knobs that matter are how
many forward passes we make (localisation, TTA), not the index structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from ..catalogue.trim import trim_to_square
from ..embed.backbone import Backbone
from ..embed.saliency import localise
from ..index.flat import FlatIndex

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class Candidate:
    item_id: int
    score: float
    rank: int


@dataclass
class MatchResult:
    candidates: list[Candidate]
    scores_all: np.ndarray | None = None      # per-item scores, for confidence features
    timings_ms: dict[str, float] = field(default_factory=dict)
    salient_area: float | None = None
    n_crops: int = 1

    def top_ids(self) -> list[int]:
        return [c.item_id for c in self.candidates]

    def as_dict(self) -> dict:
        return {
            "candidates": [asdict(c) for c in self.candidates],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "salient_area": self.salient_area,
            "n_crops": self.n_crops,
        }


class Matcher:
    """Wraps a backbone + index into a lookup.

    `localise` and `tta` each cost one extra forward pass, so the three presets
    (fast / balanced / quality) differ mainly in how many passes they make.
    """

    def __init__(
        self,
        backbone: Backbone,
        index: FlatIndex,
        localise_query: bool = True,
        tta: bool = True,
        trim_query: bool = False,
        whitener=None,
    ):
        self.backbone = backbone
        self.index = index
        self.localise_query = localise_query
        self.tta = tta
        self.trim_query = trim_query
        # If the catalogue was whitened, the query must pass through the SAME
        # transform or the two live in different spaces. Fitted on catalogue
        # embeddings only, so it leaks nothing about the query.
        self.whitener = whitener
        if whitener is not None and whitener.components_.shape[0] != index.dim:
            raise ValueError(
                f"whitener outputs {whitener.components_.shape[0]} dims but the "
                f"index has {index.dim}")
        if whitener is None and backbone.dim != index.dim:
            raise ValueError(
                f"backbone emits {backbone.dim} dims but the index has {index.dim}; "
                "the index was probably built with whitening -- pass the whitener")

    def _crops(self, img: Image.Image) -> tuple[list[Image.Image], float | None]:
        """Build the crop set. Order matters only for reporting."""
        crops = [img]
        area = None
        if self.localise_query:
            cropped, _box, area = localise(self.backbone, img)
            crops.append(cropped)
            if self.tta:
                # A wider crop hedges against the localiser cutting the object,
                # and a mirror handles the opposite-foot case.
                crops.append(cropped.transpose(Image.FLIP_LEFT_RIGHT))
        elif self.tta:
            crops.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        return crops, area

    def lookup(self, img: Image.Image, k: int = 5) -> MatchResult:
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        t = time.perf_counter()
        img = img.convert("RGB")
        if self.trim_query:
            img = trim_to_square(img)[0]
        crops, area = self._crops(img)
        timings["preprocess_localise"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        batch = torch.stack([self.backbone.transform(c) for c in crops])
        q = self.backbone.encode_tensor(batch).numpy()
        if self.whitener is not None:
            q = self.whitener.transform(q)
        timings["backbone"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        # Max over crops per item: a genuine match wins from at least one crop,
        # and crop disagreement is itself a refusal signal recorded downstream.
        best = None
        for vec in q:
            sr = self.index.search(vec, k=k, return_all=True)
            best = sr.all_item_scores if best is None else np.maximum(best, sr.all_item_scores)
        kk = min(k, self.index.n_items)
        top = np.argpartition(-best, kk - 1)[:kk]
        top = top[np.argsort(-best[top])]
        timings["search"] = (time.perf_counter() - t) * 1000

        timings["total"] = (time.perf_counter() - t_all) * 1000
        return MatchResult(
            candidates=[
                Candidate(item_id=int(self.index.unique_items[i]), score=float(best[i]), rank=r)
                for r, i in enumerate(top)
            ],
            scores_all=best,
            timings_ms=timings,
            salient_area=area,
            n_crops=len(crops),
        )

    def lookup_path(self, path: str | Path, k: int = 5) -> MatchResult:
        return self.lookup(Image.open(path), k=k)
