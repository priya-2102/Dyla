"""Exact flat index over catalogue *views*.

Measured on this machine, brute-force cosine over the whole catalogue costs
well under a millisecond, i.e. ~1% of a lookup -- the backbone forward pass is
the entire latency budget. So this stays exact numpy, and an ANN structure is
only introduced where it is actually needed (see index/ann.py for the 100k
tier). Reaching for FAISS at 25k vectors optimises the wrong term by two orders
of magnitude.

Views, not items, are the index rows: a query is one viewpoint, so collapsing an
item's views into a centroid smears the sole/side/top shots together and throws
away which view actually matched -- which is diagnostic signal we want.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cap_views(
    emb: np.ndarray,
    item_ids: np.ndarray,
    k: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Keep at most `k` maximally-diverse views per item; return row indices.

    This is not (only) a size optimisation -- it removes a bias that would
    otherwise corrupt the refusal threshold. Scoring an item by the max over its
    views makes E[score] increase with the number of views, because the max of
    k draws grows with k. Items pictured from 6 angles would then outscore items
    pictured from 2 *even when neither matches*, injecting a per-item prior into
    the very quantity we threshold on. A uniform view count removes it by
    construction.

    Diversity selection is greedy max-min cosine distance, seeded at the view
    closest to the item's centroid.
    """
    keep: list[int] = []
    order = np.argsort(item_ids, kind="stable")
    bounds = np.flatnonzero(np.diff(item_ids[order])) + 1
    for group in np.split(order, bounds):
        if len(group) <= k:
            keep.extend(group.tolist())
            continue
        vecs = emb[group]
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        chosen = [int(np.argmax(vecs @ centroid))]
        while len(chosen) < k:
            # Pick the view furthest (in cosine) from everything already chosen.
            sims = vecs @ vecs[chosen].T          # (n_views, n_chosen)
            worst = sims.max(axis=1)
            worst[chosen] = np.inf
            chosen.append(int(np.argmin(worst)))
        keep.extend(group[chosen].tolist())
    return np.array(sorted(keep), dtype=np.int64)


@dataclass
class SearchResult:
    item_ids: np.ndarray      # (k,) catalogue item ids, best first
    scores: np.ndarray        # (k,) max-over-views cosine
    view_rows: np.ndarray     # (k,) index row of the winning view, for diagnosis
    all_item_scores: np.ndarray | None = None   # (n_items,) for confidence features


class FlatIndex:
    def __init__(self, emb: np.ndarray, item_ids: np.ndarray, view_ids: np.ndarray | None = None):
        if emb.ndim != 2 or len(emb) != len(item_ids):
            raise ValueError("emb must be (N, D) and align with item_ids")
        self.emb = np.ascontiguousarray(emb.astype(np.float32))
        self.item_ids = np.asarray(item_ids)
        self.view_ids = view_ids if view_ids is not None else np.zeros(len(emb), dtype=np.int32)
        # Dense 0..M-1 item codes so aggregation is a flat scatter, not a dict.
        self.unique_items, self.item_code = np.unique(self.item_ids, return_inverse=True)
        self.n_items = len(self.unique_items)

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    def search(self, q: np.ndarray, k: int = 5, return_all: bool = False) -> SearchResult:
        """Query with a single L2-normalised vector."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        view_scores = self.emb @ q

        # Max-over-views, keeping the argmax row so we know which view won.
        best = np.full(self.n_items, -np.inf, dtype=np.float32)
        best_row = np.zeros(self.n_items, dtype=np.int64)
        np.maximum.at(best, self.item_code, view_scores)
        # np.maximum.at gives the value but not the row; recover rows by
        # matching each view against its item's winning score.
        hit = view_scores >= best[self.item_code]
        rows = np.flatnonzero(hit)
        best_row[self.item_code[rows]] = rows

        kk = min(k, self.n_items)
        top = np.argpartition(-best, kk - 1)[:kk]
        top = top[np.argsort(-best[top])]
        return SearchResult(
            item_ids=self.unique_items[top],
            scores=best[top],
            view_rows=best_row[top],
            all_item_scores=best if return_all else None,
        )
