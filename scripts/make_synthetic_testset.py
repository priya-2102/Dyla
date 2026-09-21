"""Generate a synthetic stand-in for the Part B hand-shot set.

This is NOT a substitute for the 100 hand-shot photos the brief asks for, and
the write-up says so plainly. It exists for two real reasons:

1. **The harness must be runnable from a clean checkout**, before anyone has
   picked up a phone. Every table in the report is produced by the same code
   path the real photos will use.
2. **It is the automated stumper.** Conditions are sampled with realistic
   co-occurrence -- low light genuinely causes motion blur, because a darker
   scene forces a longer exposure -- so the generated set reproduces the
   confounding that makes the real per-condition analysis hard, instead of
   handing the analysis a clean orthogonal design it will never see in reality.

Out-of-catalogue photos come from items whose ENTIRE style cluster is excluded
from the index, so a colourway twin cannot rescue them. Measured on this
catalogue, 36% of naively held-out items still had a twin indexed, which is what
drove an earlier bake-off's AUROC below chance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpm.catalogue.clusters import style_clusters
from vpm.eval.corrupt import apply_many
from vpm.eval.manifest import HEADER
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Conditions we can synthesise, mapped to the manifest vocabulary.
SYNTH = ["low_light", "motion_blur", "defocus", "off_angle",
         "partial_occlusion", "cluttered_background", "specular_reflection",
         "small_in_frame"]

# Realistic co-occurrence: a dark scene forces a longer exposure, so low light
# drags motion blur along with it. Modelling this matters -- an orthogonal design
# would make the marginal-effects analysis look far easier than reality.
COUPLED = {
    "low_light": [("motion_blur", 0.55), ("defocus", 0.20)],
    "small_in_frame": [("cluttered_background", 0.60)],
    "cluttered_background": [("off_angle", 0.30)],
    "specular_reflection": [("harsh_flash_like", 0.0)],
}


def sample_conditions(rng: np.random.Generator, n_max: int = 3) -> list[tuple[str, int]]:
    primary = str(rng.choice(SYNTH))
    chosen = {primary: int(rng.integers(2, 6))}
    for name, prob in COUPLED.get(primary, []):
        if name in SYNTH and rng.random() < prob and len(chosen) < n_max:
            chosen[name] = int(rng.integers(1, 4))
    while len(chosen) < n_max and rng.random() < 0.25:
        extra = str(rng.choice(SYNTH))
        chosen.setdefault(extra, int(rng.integers(1, 4)))
    return list(chosen.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=Path("data/items_tier1.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--out", type=Path, default=Path("data/testset_synthetic"))
    ap.add_argument("--n-items", type=int, default=40, help="distinct items to 'photograph'")
    ap.add_argument("--per-item", type=int, default=3, help="hard photos per item")
    ap.add_argument("--n-ooc", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [json.loads(l) for l in args.items.open() if l.strip()]
    have = []
    for r in rows:
        views = [v for v in range(5) if image_path(args.images, r["product_id"], v).exists()]
        if len(views) >= 3:
            have.append((r["product_id"], views))
    if len(have) < args.n_items + args.n_ooc:
        raise SystemExit(f"only {len(have)} usable items; need {args.n_items + args.n_ooc}")

    style_of = style_clusters(args.items, restrict={p for p, _ in have})
    rng.shuffle(have)

    # Out-of-catalogue items: take whole style clusters so no colourway twin of an
    # OOC item is left in the index.
    ooc, ooc_styles = [], set()
    for pid, views in have:
        if len(ooc) >= args.n_ooc:
            break
        sid = style_of.get(pid, pid)
        if sid in ooc_styles and len(ooc) >= args.n_ooc:
            continue
        ooc.append((pid, views))
        ooc_styles.add(sid)
    in_cat = [(p, v) for p, v in have if style_of.get(p, p) not in ooc_styles][: args.n_items]

    (args.out / "hard").mkdir(parents=True, exist_ok=True)
    (args.out / "clean").mkdir(parents=True, exist_ok=True)
    (args.out / "ooc").mkdir(parents=True, exist_ok=True)

    manifest: list[list] = []
    n = 0
    # Item-disjoint dev/test split, fixed here before any model sees a photo.
    dev_items = {p for p, _ in in_cat[: max(1, args.n_items // 4)]}

    for pid, views in in_cat:
        split = "dev" if pid in dev_items else "test"
        # Clean control: the ceiling, needed to measure "the gap between the halves".
        src = Image.open(image_path(args.images, pid, views[0])).convert("RGB")
        fn = f"clean/{pid}.jpg"
        src.save(args.out / fn, quality=95)
        manifest.append([f"c{n:04d}", fn, pid, "clean", split, "", "false", "exact", ""])
        n += 1
        for j in range(args.per_item):
            # Query view is held out from the one used as the clean control where
            # possible, so a hard photo is never a corrupted copy of the control.
            v = views[1 + (j % max(1, len(views) - 1))]
            raw = Image.open(image_path(args.images, pid, v)).convert("RGB")
            specs = sample_conditions(rng)
            img = apply_many(raw, specs, seed=int(rng.integers(0, 1 << 30)))
            fn = f"hard/{pid}_{j}.jpg"
            img.save(args.out / fn, quality=92)
            manifest.append([f"h{n:04d}", fn, pid, "hard", split,
                             "|".join(c for c, _ in specs), "false", "exact",
                             ";".join(f"{c}:{s}" for c, s in specs)])
            n += 1

    for pid, views in ooc:
        raw = Image.open(image_path(args.images, pid, views[0])).convert("RGB")
        specs = sample_conditions(rng)
        img = apply_many(raw, specs, seed=int(rng.integers(0, 1 << 30)))
        fn = f"ooc/{pid}.jpg"
        img.save(args.out / fn, quality=92)
        manifest.append([f"o{n:04d}", fn, "", "out_of_catalogue", "test",
                         "|".join(c for c, _ in specs), "false", "exact",
                         f"style cluster {style_of.get(pid, pid)} excluded from index"])
        n += 1

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(manifest)

    excl = sorted({p for p, _ in ooc} | {p for p, _ in have
                                         if style_of.get(p, p) in ooc_styles})
    (args.out / "exclude_from_index.json").write_text(json.dumps(excl))

    print(f"wrote {len(manifest)} rows -> {args.out}/manifest.csv")
    print(f"  {args.n_items} in-catalogue items x ({args.per_item} hard + 1 clean)")
    print(f"  {len(ooc)} out-of-catalogue items; {len(excl)} product ids excluded from the index")
    print(f"  dev items: {len(dev_items)} (item-disjoint from test)")


if __name__ == "__main__":
    main()
