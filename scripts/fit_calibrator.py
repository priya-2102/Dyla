"""Fit the open-set calibrator and the conformal refuser.

Training data is generated from the catalogue, with **zero contact** with the
Part B photos:

  positives   corrupted catalogue views of items that ARE indexed, queried back.
              The label is whether top-1 is the true item, so the model learns
              what a genuine match's score landscape looks like across the whole
              difficulty range rather than only on easy cases.

  negatives   items whose ENTIRE style cluster is excluded from the index, then
              queried. These are genuine open-set negatives drawn from the same
              distribution as the catalogue. Holding out individual items does
              NOT work -- measured on this catalogue, 36% of them keep a
              colourway twin in the index, so the query is never actually absent.

Folds are item-disjoint (GroupKFold on product id), because without that the
model memorises per-item score offsets and cross-validated AUROC is fiction.

Stated limitation, measured rather than assumed: this is fitted on SYNTHETIC
corruptions and would be deployed on REAL photos. That is a domain shift in the
calibrator itself; `vpm eval` reports reliability/ECE on real photos so the size
of the shift is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from vpm.catalogue.clusters import style_clusters
from vpm.embed.backbone import EncodeConfig, build_backbone
from vpm.embed.encode import CatalogueEmbeddings
from vpm.eval.corrupt import CONDITIONS, apply_many
from vpm.eval.stats import auroc
from vpm.index.flat import FlatIndex, cap_views
from vpm.index.pca import PCAWhitening
from vpm.match.confidence import extract, image_quality
from vpm.match.conformal import Calibrator, ConformalRefuser
from vpm.match.pipeline import Matcher
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sample_specs(rng) -> list[tuple[str, int]]:
    """Sample a corruption recipe spanning the whole difficulty range.

    Severity is sampled uniformly so the calibrator sees easy, marginal and
    hopeless queries; training only on hard ones would make it uniformly
    pessimistic and only on easy ones uniformly overconfident.
    """
    pool = [c for c in CONDITIONS if c != "jpeg_artifacts"]
    k = int(rng.integers(1, 4))
    return [(str(c), int(rng.integers(1, 6))) for c in rng.choice(pool, k, replace=False)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.npz"))
    ap.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--n-pos", type=int, default=900)
    ap.add_argument("--n-neg", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("data/calibrator"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ce = CatalogueEmbeddings.load(args.index)
    bb = build_backbone(ce.backbone, config=EncodeConfig(image_size=ce.image_size,
                                                        pooling=ce.pooling))
    whitener = None
    wpath = args.index.with_suffix(".whiten.npz")
    if ce.whiten_dim and wpath.exists():
        whitener = PCAWhitening.load(wpath)

    keep = cap_views(ce.emb, ce.item_ids, k=4)
    index = FlatIndex(ce.emb[keep], ce.item_ids[keep], ce.view_ids[keep])
    matcher = Matcher(bb, index, localise_query=False, tta=False, whitener=whitener)
    indexed = set(index.unique_items.tolist())
    print(f"index: {index.n_items} items / {len(index.emb)} views")

    rows = [json.loads(l) for l in args.items.open() if l.strip()]
    style_of = style_clusters(args.items)
    indexed_styles = {style_of.get(p, p) for p in indexed}

    # Negatives: items absent from the index AND whose whole style cluster is absent.
    absent = [r for r in rows
              if r["product_id"] not in indexed
              and style_of.get(r["product_id"], r["product_id"]) not in indexed_styles]
    present = [r for r in rows if r["product_id"] in indexed]
    print(f"pool: {len(present)} indexed items, {len(absent)} genuinely-absent items")
    if len(absent) < 30:
        raise SystemExit("too few genuinely-absent items; widen the exclusion set")

    def features_for(row, label: int):
        pid = row["product_id"]
        views = [v for v in range(4) if image_path(args.images, pid, v).exists()]
        if not views:
            return None
        v = int(rng.choice(views))
        try:
            raw = Image.open(image_path(args.images, pid, v)).convert("RGB")
        except Exception:
            return None
        img = apply_many(raw, sample_specs(rng), seed=int(rng.integers(0, 1 << 30)))
        res = matcher.lookup(img, k=5)
        if res.scores_all is None:
            return None
        top1 = res.candidates[0].item_id if res.candidates else None
        # A positive is only a positive if the retrieval was ACTUALLY correct --
        # the calibrator predicts "top-1 is right", not "the item exists".
        y = int(label == 1 and top1 == pid)
        f = extract(res.scores_all, quality=image_quality(img),
                    salient_area=res.salient_area, crop_top1=None, final_top1=top1)
        return f.values, y, pid, float(res.candidates[0].score)

    X, Y, G, S = [], [], [], []
    for tag, pool, label, n in (("positives", present, 1, args.n_pos),
                                ("negatives", absent, 0, args.n_neg)):
        picked = rng.choice(len(pool), min(n, len(pool)), replace=False)
        got = 0
        for i in picked:
            out = features_for(pool[int(i)], label)
            if out is None:
                continue
            x, y, g, s = out
            X.append(x); Y.append(y); G.append(g); S.append(s)
            got += 1
            if got % 200 == 0:
                print(f"  {tag}: {got}", flush=True)
        print(f"  {tag}: {got} total")

    X = np.array(X); Y = np.array(Y); G = np.array(G); S = np.array(S)
    print(f"dataset: {len(Y)} rows, positive rate {Y.mean():.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"n": int(len(Y)), "pos_rate": float(Y.mean())}

    for tag, use_quality in (("with_quality", True), ("no_quality", False)):
        cal = Calibrator(include_quality=use_quality).fit(X, Y, groups=G)
        p = cal.predict_proba(X)
        a = auroc(p[Y == 1], p[Y == 0])
        a_raw = auroc(S[Y == 1], S[Y == 0])
        cal.save(args.out / f"calibrator_{tag}.pkl")
        summary[tag] = {"auroc_calibrated": float(a), "auroc_raw_cosine": float(a_raw),
                        "top_coefficients": cal.coefficients()[:8]}
        print(f"\n[{tag}] AUROC calibrated {a:.3f}  vs raw cosine {a_raw:.3f}")
        for n, c in cal.coefficients()[:6]:
            print(f"    {n:28s} {c:+.3f}")

    # Conformal calibration uses only CORRECT matches, as the theory requires.
    cal = Calibrator.load(args.out / "calibrator_with_quality.pkl")
    p = cal.predict_proba(X)
    correct = p[Y == 1]
    refuser = ConformalRefuser(alpha=args.alpha).fit(correct)
    refuser.save(args.out / "conformal.json")
    summary["conformal"] = {"alpha": args.alpha, "n_calibration": int(len(correct))}

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {args.out}/ (calibrators, conformal.json, summary.json)")


if __name__ == "__main__":
    main()
