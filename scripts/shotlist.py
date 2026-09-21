"""Resolve your physical shoes to catalogue ids, then emit a shot plan.

This is step 1 of Part B and it must happen BEFORE the camera comes out. The
failure mode it prevents: shoot 130 photos, then discover a third of the SKUs
are not in the catalogue and the test set is unsalvageable without reshooting.

Usage
-----
    # 1. list what you physically have, one per line: "brand  model words"
    cat > data/my_shoes.txt <<'EOF'
    Puma Softride Enzo
    Campus North Plus
    Nike Revolution 6
    EOF

    # 2. resolve against the catalogue
    python scripts/shotlist.py --have data/my_shoes.txt --items data/items.jsonl

Emits `data/shotlist.csv`: for each shoe, the candidate catalogue ids with
brand/name/colour so you can eyeball the right one, plus a per-item plan of which
failure conditions to induce. Items that resolve to nothing are listed separately
so they leave the shoot list rather than poisoning the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpm.eval.manifest import CONDITIONS

# A shoot plan that produces the co-occurrence structure the analysis needs.
# Single-condition photos are what make marginal effects identifiable at all; if
# every photo stacks three conditions, no regression can separate them.
PLAN = [
    ("clean", []),
    ("hard", ["low_light"]),
    ("hard", ["motion_blur"]),
    ("hard", ["off_angle"]),
    ("hard", ["partial_occlusion", "hand_or_foot_in_frame"]),
    ("hard", ["cluttered_background", "small_in_frame"]),
    ("hard", ["specular_reflection"]),
    ("hard", ["low_light", "motion_blur"]),
]

_WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--have", type=Path, required=True, help="one shoe per line: brand + model")
    ap.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    ap.add_argument("--top", type=int, default=5, help="candidate ids to show per shoe")
    ap.add_argument("--out", type=Path, default=Path("data/shotlist.csv"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.items.open() if l.strip()]
    index = [(r, toks(r["brand"]) | toks(r["name"])) for r in rows]
    print(f"catalogue: {len(rows)} items")

    wanted = [l.strip() for l in args.have.read_text().splitlines() if l.strip()]
    found, missing = [], []

    for line in wanted:
        q = toks(line)
        if not q:
            continue
        scored = []
        for r, t in index:
            overlap = len(q & t)
            if overlap == 0:
                continue
            # Jaccard-ish: reward covering the query, lightly penalise long names.
            scored.append((overlap / len(q) - 0.02 * max(0, len(t) - len(q)), r))
        scored.sort(key=lambda kv: -kv[0])
        top = scored[: args.top]
        if not top or top[0][0] < 0.5:
            missing.append(line)
            continue
        found.append((line, top))

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["your_shoe", "rank", "score", "product_id", "brand", "name",
                    "colour", "url"])
        for line, top in found:
            for i, (sc, r) in enumerate(top, 1):
                w.writerow([line, i, round(sc, 3), r["product_id"], r["brand"],
                            r["name"], r.get("base_colour", ""), r["source_url"]])

    plan_path = args.out.with_name("shot_plan.md")
    with plan_path.open("w", encoding="utf-8") as fh:
        fh.write("# Shot plan\n\n")
        fh.write(f"{len(found)} shoes resolved. Target: **{len(PLAN)} photos each** "
                 f"= {len(found) * len(PLAN)} total "
                 f"({len(found)} clean controls + {len(found) * (len(PLAN)-1)} hard).\n\n")
        fh.write("Shoot **more than you need** and lock the split afterwards: 130 hard "
                 "+ 30 out-of-catalogue, of which 100 + 20 become the locked test set and "
                 "the rest is dev for threshold selection. Dev and test must be "
                 "**item-disjoint**, not merely photo-disjoint.\n\n")
        fh.write("Per shoe:\n\n")
        for kind, conds in PLAN:
            fh.write(f"- `{kind}` — {' + '.join(conds) if conds else 'well lit, plain background, fills ~70% of frame'}\n")
        fh.write("\n## Why single-condition shots matter\n\n")
        fh.write("Half the plan is single-condition on purpose. If every photo stacks "
                 "three conditions, the marginal effect of each is not identifiable and "
                 "the per-condition analysis collapses into one number.\n\n")
        fh.write("## Also record\n\n")
        fh.write("- `occludes_logo` when a hand/foot covers the brand mark — a hand over "
                 "the logo is categorically worse than the same area of midsole, and no "
                 "area-based severity captures it.\n")
        fh.write("- `sku_confidence`: `exact` if you confirmed the SKU from the box/tongue "
                 "label, `style_match` if the model matches but your colourway is not "
                 "listed, `uncertain` otherwise. **`style_match` rows are genuinely "
                 "out-of-catalogue at SKU level and make the hardest refusal negatives.**\n")
        fh.write(f"\nCondition vocabulary: `{'`, `'.join(CONDITIONS)}`\n")

    print(f"\nresolved {len(found)} / {len(wanted)} shoes -> {args.out}")
    print(f"shot plan -> {plan_path}")
    if missing:
        print(f"\nNOT FOUND ({len(missing)}) — drop these before shooting, or widen the scrape:")
        for m in missing:
            print(f"   {m}")
    if found:
        print(f"\ntarget: {len(found) * len(PLAN)} photos "
              f"({len(found)} clean + {len(found)*(len(PLAN)-1)} hard)")


if __name__ == "__main__":
    main()
