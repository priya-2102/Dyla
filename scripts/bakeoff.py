"""Backbone bake-off on our own catalogue. Spends no test-set budget.

Protocol, per backbone:
  * index views 1..N of each item (trimmed, view-capped)
  * query with view 0, held out -- clean, and corrupted to stand in for a phone photo
  * hold a disjoint block of items OUT of the index entirely; their view-0
    queries are genuine open-set negatives

Scored on the JOINT criterion of ranking (Recall@1/@5) and separability
(AUROC of the top-1 score, in-catalogue vs held-out). A backbone can win ranking
and lose separability, and separability is what the refusal extension rests on,
so picking on Recall@1 alone would be picking on half the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpm.catalogue.clusters import style_clusters
from vpm.catalogue.trim import trim_to_square
from vpm.embed.backbone import EncodeConfig, build_backbone
from vpm.eval.corrupt import apply_many
from vpm.index.flat import FlatIndex, cap_views
from vpm.index.pca import PCAWhitening
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# A fixed, reproducible "hard phone photo" recipe. Deliberately multi-condition:
# real adverse photos co-occur, and a single-condition proxy would flatter the
# system relative to the hand-shot set it is meant to predict.
HARD_RECIPE = [("low_light", 2), ("motion_blur", 2), ("cluttered_background", 2)]


def available_views(root: Path, pid: int, max_views: int = 5) -> list[int]:
    return [v for v in range(max_views) if image_path(root, pid, v).exists()]


def load_raw(root: Path, pid: int, view: int) -> Image.Image | None:
    try:
        img = Image.open(image_path(root, pid, view))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def load(root: Path, pid: int, view: int, size: int, trim: bool = True) -> Image.Image | None:
    p = image_path(root, pid, view)
    try:
        img = Image.open(p)
        img.load()
    except Exception:
        return None
    if trim:
        return trim_to_square(img, size=size)[0]
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def encode(bb, images: list[Image.Image], bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), bs):
        batch = torch.stack([bb.transform(im) for im in images[i : i + bs]])
        out.append(bb.encode_tensor(batch).numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, bb.dim), np.float32)


def run(name: str, items: list[dict], root: Path, n_heldout: int, size: int, pooling: str,
        style_of: dict[int, int] | None = None, whiten_dim: int | None = None) -> dict:
    t0 = time.time()
    bb = build_backbone(name, config=EncodeConfig(image_size=size, pooling=pooling))
    size = bb.cfg.image_size  # may be forced to the model's native resolution

    # Hold out whole STYLE CLUSTERS, not individual items: 36% of singly
    # held-out items still had a colourway twin in the index, which made the
    # "absent" queries anything but absent and drove AUROC below chance.
    style_of = style_of or {}
    order = np.random.default_rng(0).permutation(len(items))
    items = [items[i] for i in order]
    held, indexed, held_styles = [], [], set()
    for it in items:
        sid = style_of.get(it["product_id"], it["product_id"])
        if len(held) < n_heldout or sid in held_styles:
            held.append(it)
            held_styles.add(sid)
        else:
            indexed.append(it)
    indexed = [it for it in indexed if style_of.get(it["product_id"], it["product_id"]) not in held_styles]

    # ---- index: views 1..N ----
    imgs, item_ids = [], []
    for it in indexed:
        for v in it["views"][1:]:
            im = load(root, it["product_id"], v, size)
            if im is not None:
                imgs.append(im)
                item_ids.append(it["product_id"])
    emb = encode(bb, imgs)
    item_ids = np.array(item_ids)

    # Whitening is fitted on the INDEX side only -- never on query embeddings --
    # so it carries no information about the queries it will be scored against.
    whitener = None
    if whiten_dim:
        whitener = PCAWhitening(dim=whiten_dim).fit(emb)
        emb = whitener.transform(emb)

    keep = cap_views(emb, item_ids, k=4)
    index = FlatIndex(emb[keep], item_ids[keep])
    t_index = time.time() - t0

    # ---- queries: view 0, clean and corrupted ----
    def queries(pool, corrupt: bool):
        qs, truth = [], []
        for it in pool:
            if corrupt:
                # Corrupt at native resolution, THEN downsample -- real adverse
                # conditions are optical and precede the sensor's downsampling.
                raw = load_raw(root, it["product_id"], it["views"][0])
                if raw is None:
                    continue
                im = apply_many(raw, HARD_RECIPE, seed=it["product_id"])
                im = trim_to_square(im, size=size)[0]
            else:
                im = load(root, it["product_id"], it["views"][0], size)
                if im is None:
                    continue
            qs.append(im)
            truth.append(it["product_id"])
        Z = encode(bb, qs)
        if whitener is not None and len(Z):
            Z = whitener.transform(Z)
        return Z, np.array(truth)

    res = {"backbone": name, "dim": bb.dim, "whiten_dim": whiten_dim,
           "image_size": size, "pooling": pooling,
           "n_index_rows": int(len(keep)), "n_items": int(index.n_items),
           "index_seconds": round(t_index, 1)}

    for tag, corrupt in (("clean", False), ("hard", True)):
        Q, truth = queries(indexed, corrupt)
        r1 = r5 = s_r1 = s_r5 = 0
        top1 = np.zeros(len(Q), np.float32)
        sty = lambda p: style_of.get(int(p), int(p))
        for i, q in enumerate(Q):
            sr = index.search(q, k=5)
            top1[i] = sr.scores[0]
            if sr.item_ids[0] == truth[i]:
                r1 += 1
            if truth[i] in sr.item_ids:
                r5 += 1
            # Style level: credit the right model in the wrong colourway, which
            # at SKU level is an unanswerable question on many photos.
            if sty(sr.item_ids[0]) == sty(truth[i]):
                s_r1 += 1
            if sty(truth[i]) in {sty(x) for x in sr.item_ids}:
                s_r5 += 1
        n = max(len(Q), 1)
        res[f"{tag}_r1"] = round(r1 / n, 4)
        res[f"{tag}_r5"] = round(r5 / n, 4)
        res[f"{tag}_style_r1"] = round(s_r1 / n, 4)
        res[f"{tag}_style_r5"] = round(s_r5 / n, 4)
        res[f"{tag}_n"] = int(n)

        # open-set separability: held-out items are absent from the index
        Qo, _ = queries(held, corrupt)
        neg = np.array([index.search(q, k=1).scores[0] for q in Qo], np.float32)
        if len(neg) and len(top1):
            lab = np.r_[np.ones(len(top1)), np.zeros(len(neg))]
            sc = np.r_[top1, neg]
            order = np.argsort(sc)
            ranks = np.empty(len(sc), float)
            ranks[order] = np.arange(1, len(sc) + 1)
            npos, nneg = lab.sum(), (1 - lab).sum()
            auroc = (ranks[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            res[f"{tag}_auroc"] = round(float(auroc), 4)
            res[f"{tag}_pos_mean"] = round(float(top1.mean()), 4)
            res[f"{tag}_neg_mean"] = round(float(neg.mean()), 4)
    res["total_seconds"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=Path("data/items_tier1.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--n-items", type=int, default=1200)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pooling", default="cls_patch")
    ap.add_argument("--backbones", nargs="+",
                    default=["dinov2-small-reg", "dinov2-base", "siglip2-base"])
    ap.add_argument("--whiten", nargs="+", type=int, default=[0],
                    help="PCA-whitening dims to sweep; 0 means no whitening")
    ap.add_argument("--out", type=Path, default=Path("reports/bakeoff.json"))
    args = ap.parse_args()

    pool = []
    with args.items.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            views = available_views(args.images, row["product_id"])
            if len(views) >= 3:      # need >=1 query view and >=2 index views
                pool.append({"product_id": row["product_id"], "views": views})
            if len(pool) >= args.n_items:
                break
    print(f"pool: {len(pool)} items with >=3 downloaded views "
          f"({args.n_heldout} held out of the index as open-set negatives)")

    style_of = style_clusters(args.items, restrict={p["product_id"] for p in pool})
    n_styles = len(set(style_of.values()))
    print(f"style clusters: {n_styles} over {len(style_of)} items "
          f"({len(style_of)/max(n_styles,1):.2f} SKUs/style)")

    rows = []
    for name in args.backbones:
        for wd in args.whiten:
            r = run(name, pool, args.images, args.n_heldout, args.size, args.pooling,
                    style_of, whiten_dim=(wd or None))
            rows.append(r)
            print(json.dumps(r), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    hdr = (f"{'backbone':18s} {'wht':>5s} {'cln R@1':>8s} {'cln R@5':>8s} {'clnSty@1':>9s} "
           f"{'hard R@1':>9s} {'hard R@5':>9s} {'hardSty@1':>10s} {'clnAUC':>7s} {'hardAUC':>8s}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['backbone']:18s} {str(r['whiten_dim'] or '-'):>5s} "
              f"{r['clean_r1']:8.3f} {r['clean_r5']:8.3f} {r['clean_style_r1']:9.3f} "
              f"{r['hard_r1']:9.3f} {r['hard_r5']:9.3f} {r['hard_style_r1']:10.3f} "
              f"{r['clean_auroc']:7.3f} {r['hard_auroc']:8.3f}")


if __name__ == "__main__":
    main()
