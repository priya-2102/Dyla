"""Style clusters over the colourway graph.

Two jobs, both load-bearing:

1. **Ground truth is ambiguous at SKU level.** "Puma Softride, Black/Red" and
   "Puma Softride, Triple Black" are separate SKUs whose photos differ in a
   panel or two. Asking a matcher to pick the right one from a dark phone photo
   is sometimes genuinely unanswerable, so accuracy is reported at BOTH SKU and
   style level, and "correct style, wrong colourway" becomes its own error class
   rather than an unexplained miss.

2. **Naive item hold-out does not produce open-set negatives.** Measured on our
   catalogue: 36% of individually held-out items still had a colourway variant
   sitting in the index, so the "absent" query had a near-twin to match against.
   That alone drove bake-off AUROC below 0.5 -- worse than chance. Genuine
   negatives require holding out the whole cluster.
"""

from __future__ import annotations

import json
from pathlib import Path


class _DSU:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def style_clusters(items_path: Path, restrict: set[int] | None = None) -> dict[int, int]:
    """Map product_id -> style_id (the cluster's smallest product id).

    Edges come from the catalogue's own `colours` graph, which lists the same
    model in other colourways. Only edges whose endpoints are both present are
    used, so the clustering reflects the catalogue we actually hold.
    """
    rows = []
    with items_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    present = {r["product_id"] for r in rows}
    if restrict is not None:
        present &= restrict

    dsu = _DSU()
    for r in rows:
        pid = r["product_id"]
        if pid not in present:
            continue
        dsu.find(pid)
        for v in r.get("colour_variants", []):
            if v in present:
                dsu.union(pid, v)

    # Name each cluster by its smallest member so ids are stable across runs.
    groups: dict[int, list[int]] = {}
    for pid in present:
        groups.setdefault(dsu.find(pid), []).append(pid)
    return {pid: min(members) for members in groups.values() for pid in members}
