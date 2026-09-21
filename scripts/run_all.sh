#!/usr/bin/env bash
# Full reproduction, clean checkout to report. Every table in REPORT.md comes
# from here. Safe to re-run: scraping and downloads are resumable and skip work
# already done.
set -euo pipefail

ITEMS=${ITEMS:-data/items.jsonl}
IMAGES=${IMAGES:-data/images}
LIMIT=${LIMIT:-12000}          # catalogue items; the brief requires >=5000 images
BACKBONE=${BACKBONE:-siglip2-base}

echo "==> 1/7 harvest footwear product URLs from the public sitemaps"
vpm scrape sitemaps --out data/refs_footwear.jsonl

echo "==> 2/7 fetch product pages (rate-limited)"
vpm scrape products --refs data/refs_footwear.jsonl --out "$ITEMS" --limit "$LIMIT"

echo "==> 3/7 download catalogue images (resumable)"
vpm scrape images --items "$ITEMS" --images "$IMAGES" --max-views 4

echo "==> 4/7 build the synthetic stand-in test set (NOT the Part B photos)"
python scripts/make_synthetic_testset.py --items "$ITEMS" --images "$IMAGES" \
       --n-items 40 --per-item 3 --n-ooc 20

echo "==> 5/7 encode + whiten the catalogue index"
vpm index --backbone "$BACKBONE" --whiten 256 --max-views 4 \
          --items "$ITEMS" --images "$IMAGES" \
          --exclude data/testset_synthetic/exclude_from_index.json \
          --out data/index.npz

echo "==> 6/7 fit the open-set calibrator and conformal refuser"
python scripts/fit_calibrator.py --index data/index.npz --items "$ITEMS" --images "$IMAGES"

echo "==> 7/7 evaluate"
vpm eval --index data/index.npz --items "$ITEMS" \
         --testset data/testset_synthetic --out reports/eval

echo
echo "done. see reports/eval/report.md"
