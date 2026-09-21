"""vpm -- visual product matcher CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _scrape(args):
    from .catalogue.build import (download_catalogue_images, fetch_items,
                                  harvest_refs)
    from .scrape.sitemap import FOOTWEAR_CATEGORIES

    if args.stage == "sitemaps":
        n = harvest_refs(Path(args.out), n_sitemaps=args.n_sitemaps,
                         categories=None if args.all_categories else FOOTWEAR_CATEGORIES,
                         limit=args.limit)
        print(f"harvested {n} product refs -> {args.out}")
    elif args.stage == "products":
        n = fetch_items(Path(args.refs), Path(args.out), workers=args.workers,
                        delay=args.delay, limit=args.limit)
        print(f"wrote {n} catalogue records -> {args.out}")
    elif args.stage == "images":
        res = download_catalogue_images(Path(args.items), Path(args.images),
                                        max_views=args.max_views, workers=args.workers)
        print(f"images: {res}")


def _eval(args):
    import warnings
    warnings.filterwarnings("ignore")
    from .catalogue.clusters import style_clusters
    from .eval.harness import evaluate, run_photos, save, to_markdown
    from .eval.manifest import check_split_disjoint, load_manifest, summarise

    matcher, _ce = _load_matcher(args)
    catalogue_ids = set(matcher.index.unique_items.tolist())
    root = Path(args.testset)
    photos = load_manifest(root / "manifest.csv", photo_root=root)
    check_split_disjoint(photos)
    print("manifest:", json.dumps(summarise(photos), indent=2, default=str))

    calibrator = refuser = None
    cal_dir = Path(args.calibrator) if args.calibrator else None
    if cal_dir and (cal_dir / "calibrator_with_quality.pkl").exists():
        from .match.conformal import Calibrator, ConformalRefuser
        calibrator = Calibrator.load(cal_dir / "calibrator_with_quality.pkl")
        if (cal_dir / "conformal.json").exists():
            refuser = ConformalRefuser.load(cal_dir / "conformal.json")
        print(f"loaded calibrator{' + conformal refuser' if refuser else ''} from {cal_dir}")
    else:
        print("no calibrator found -- refusal metrics will fall back to raw cosine")

    style_of = style_clusters(Path(args.items)) if args.items else {}
    results = run_photos(matcher, photos, root, style_of=style_of, k=args.k,
                         calibrator=calibrator, refuser=refuser)
    print(f"ran {len(results)} photos")

    report = evaluate(results, split=args.split)
    save(report, results, Path(args.out))
    print("\n" + to_markdown(report))
    print(f"\nwrote {args.out}/report.md, report.json, per_photo.jsonl")


def _build_index(args):
    import json as _json
    from .embed.backbone import EncodeConfig, build_backbone
    from .embed.encode import encode_catalogue
    from .index.pca import PCAWhitening

    exclude = set()
    if args.exclude and Path(args.exclude).exists():
        exclude = set(_json.loads(Path(args.exclude).read_text()))
        print(f"excluding {len(exclude)} product ids from the index "
              f"(out-of-catalogue items and their colourway twins)")

    bb = build_backbone(args.backbone, config=EncodeConfig(image_size=args.size, pooling=args.pooling))
    print(f"encoding with {args.backbone} @{bb.cfg.image_size}px (dim {bb.dim}) on {bb.device}")
    t0 = time.time()
    ce = encode_catalogue(
        Path(args.items), Path(args.images), bb,
        max_views=args.max_views, trim=not args.no_trim, mirror=args.mirror,
        batch_size=args.batch, limit=args.limit, exclude=exclude,
    )
    if args.whiten:
        # Fitted on catalogue embeddings only. Measured on this catalogue it
        # widens the same-item / different-item gap ~15x, which is what the
        # refusal scorer needs even though it barely moves Recall@1.
        w = PCAWhitening(dim=args.whiten).fit(ce.emb)
        ce.emb = w.transform(ce.emb)
        ce.whiten_dim = args.whiten
        w.save(Path(args.out).with_suffix(".whiten.npz"))
        print(f"applied PCA-whitening -> {args.whiten} dims")
    ce.save(Path(args.out))
    dt = time.time() - t0
    n = len(ce.emb)
    print(f"encoded {n} views of {len(set(ce.item_ids.tolist()))} items in {dt:.1f}s "
          f"({dt / max(n, 1) * 1000:.1f} ms/view) -> {args.out}")


def _load_matcher(args):
    from .embed.backbone import EncodeConfig, build_backbone
    from .embed.encode import CatalogueEmbeddings
    from .index.flat import FlatIndex, cap_views
    from .match.pipeline import Matcher

    ce = CatalogueEmbeddings.load(Path(args.index))
    bb = build_backbone(ce.backbone, config=EncodeConfig(image_size=ce.image_size, pooling=ce.pooling))
    keep = cap_views(ce.emb, ce.item_ids, k=args.cap_views)
    idx = FlatIndex(ce.emb[keep], ce.item_ids[keep], ce.view_ids[keep])
    return Matcher(bb, idx, localise_query=not args.no_localise, tta=not args.no_tta), ce


def _query(args):
    matcher, ce = _load_matcher(args)
    names = {}
    if args.items and Path(args.items).exists():
        with open(args.items) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    names[r["product_id"]] = f"{r['brand']} — {r['name']}"
    res = matcher.lookup_path(args.image, k=args.k)
    print(f"\nquery: {args.image}")
    print(f"catalogue: {matcher.index.n_items} items / {len(matcher.index.emb)} views "
          f"({ce.backbone} @{ce.image_size}px)\n")
    for c in res.candidates:
        print(f"  {c.rank + 1}. {c.score:.4f}  {c.item_id}  {names.get(c.item_id, '')}")
    print("\ntimings (ms):", json.dumps({k: round(v, 1) for k, v in res.timings_ms.items()}))
    print(f"crops: {res.n_crops} | salient area: {res.salient_area}")


def _bench(args):
    from PIL import Image
    matcher, ce = _load_matcher(args)
    img = Image.open(args.image).convert("RGB") if args.image else Image.fromarray(
        np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8))
    for _ in range(3):
        matcher.lookup(img)
    runs = [matcher.lookup(img).timings_ms for _ in range(args.iters)]
    keys = runs[0].keys()
    print(f"\n{len(matcher.index.emb)} index rows / {matcher.index.n_items} items, "
          f"{matcher.lookup(img).n_crops} crops, {args.iters} iters\n")
    print(f"{'stage':22s} {'median':>9s} {'p90':>9s}")
    print("-" * 42)
    for k in keys:
        vals = sorted(r[k] for r in runs)
        print(f"{k:22s} {vals[len(vals)//2]:8.2f}ms {vals[int(len(vals)*0.9)]:8.2f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(prog="vpm", description="Visual product matcher")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scrape", help="build the catalogue from Myntra's public sitemaps")
    sc.add_argument("stage", choices=["sitemaps", "products", "images"])
    sc.add_argument("--out", default="data/refs_footwear.jsonl")
    sc.add_argument("--refs", default="data/refs_footwear.jsonl")
    sc.add_argument("--items", default="data/items.jsonl")
    sc.add_argument("--images", default="data/images")
    sc.add_argument("--n-sitemaps", type=int, default=None)
    sc.add_argument("--all-categories", action="store_true")
    sc.add_argument("--max-views", type=int, default=4)
    sc.add_argument("--workers", type=int, default=8)
    sc.add_argument("--delay", type=float, default=0.1)
    sc.add_argument("--limit", type=int, default=None)
    sc.set_defaults(func=_scrape)

    e = sub.add_parser("eval", help="evaluate against a labelled test set")
    e.add_argument("--index", default="data/index.npz")
    e.add_argument("--items", default="data/items.jsonl")
    e.add_argument("--testset", default="data/testset_synthetic")
    e.add_argument("--split", default="test")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--cap-views", type=int, default=4)
    e.add_argument("--no-localise", action="store_true")
    e.add_argument("--no-tta", action="store_true")
    e.add_argument("--calibrator", default="data/calibrator")
    e.add_argument("--out", default="reports/eval")
    e.set_defaults(func=_eval)

    b = sub.add_parser("index", help="encode the catalogue into an index")
    b.add_argument("--items", default="data/items.jsonl")
    b.add_argument("--images", default="data/images")
    b.add_argument("--backbone", default="dinov2-base")
    b.add_argument("--size", type=int, default=224)
    b.add_argument("--pooling", default="cls_patch")
    b.add_argument("--max-views", type=int, default=5)
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--mirror", action="store_true", help="also index a mirrored copy of each view")
    b.add_argument("--no-trim", action="store_true")
    b.add_argument("--whiten", type=int, default=0, help="PCA-whitening dims (0 = off)")
    b.add_argument("--exclude", default=None, help="JSON list of product ids to keep out")
    b.add_argument("--out", default="data/index.npz")
    b.set_defaults(func=_build_index)

    q = sub.add_parser("query", help="match a photo against the catalogue")
    q.add_argument("image")
    q.add_argument("--index", default="data/index.npz")
    q.add_argument("--items", default="data/items.jsonl")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--cap-views", type=int, default=4)
    q.add_argument("--no-localise", action="store_true")
    q.add_argument("--no-tta", action="store_true")
    q.set_defaults(func=_query)

    n = sub.add_parser("bench", help="latency breakdown for a single lookup")
    n.add_argument("--image", default=None)
    n.add_argument("--index", default="data/index.npz")
    n.add_argument("--items", default=None)
    n.add_argument("--iters", type=int, default=25)
    n.add_argument("--cap-views", type=int, default=4)
    n.add_argument("--no-localise", action="store_true")
    n.add_argument("--no-tta", action="store_true")
    n.set_defaults(func=_bench)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
