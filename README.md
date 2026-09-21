# Visual Product Matcher — footwear

Photograph a shoe, get the matching catalogue SKU back, ranked, with a
confidence score that means something — and a refusal when the shoe is not in
the catalogue at all.

Catalogue: **footwear scraped from Myntra**. A 345,331-item footwear pool was
harvested from the public sitemaps; the working catalogue is a multi-view subset
of it.

- **[REPORT.md](REPORT.md)** — the write-up: what was measured, what was rejected, and what does not work.
- **[DECISIONS.md](DECISIONS.md)** — the decision log, one entry per decision with the evidence that moved it.
- **[logs/SESSION.md](logs/SESSION.md)** — how the work actually happened, including the two places the tool's recommendation was overruled with measurement. Raw run logs in `logs/runs/`.

---

## Quick start

Requires Python 3.11+. On Apple Silicon, PyTorch MPS is used automatically; the
code falls back to CPU everywhere.

```bash
git clone <this repo> && cd Dyla
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

That installs the `vpm` CLI. To reproduce from nothing:

```bash
# 1. Harvest footwear product URLs from Myntra's sitemaps (~2 min, no page fetches)
vpm scrape sitemaps --out data/refs_footwear.jsonl

# 2. Fetch product pages -> catalogue records (rate-limited; ~10 items/s)
vpm scrape products --refs data/refs_footwear.jsonl --out data/items.jsonl --limit 12000

# 3. Download catalogue images (resumable; re-run to pick up stragglers)
vpm scrape images --items data/items.jsonl --images data/images --max-views 4

# 4. Build a synthetic stand-in test set (see REPORT.md §Part B for why)
python scripts/make_synthetic_testset.py --n-items 40 --per-item 3 --n-ooc 20

# 5. Encode the catalogue into a whitened index (~15 min on an M3)
vpm index --backbone siglip2-base --whiten 256 --max-views 4 \
          --items data/items.jsonl --images data/images \
          --exclude data/testset_synthetic/exclude_from_index.json \
          --out data/index.npz

# 6. Match a single photo
vpm query path/to/photo.jpg --index data/index.npz --items data/items.jsonl

# 7. Latency breakdown for one lookup
vpm bench --index data/index.npz --iters 25

# 8. Full evaluation: per-condition accuracy, marginal effects, refusal, gap analysis
vpm eval --index data/index.npz --items data/items.jsonl \
         --testset data/testset_synthetic --out reports/eval
```

Every table in `REPORT.md` comes from step 8. Nothing is hand-copied.

### Part B — the hand-shot test set

The 100 adversarial photographs are **not** in this repo; they require items the
author physically owns (see *Known limitations*). The tooling to produce them:

```bash
# list what you physically have, one per line: "brand model"
printf 'Puma Softride Enzo\nCampus North Plus\n' > data/my_shoes.txt

# resolve to catalogue ids BEFORE shooting -- anything unresolvable leaves the
# shoot list rather than poisoning the test set
python scripts/shotlist.py --have data/my_shoes.txt --items data/items.jsonl
```

That writes `data/shotlist.csv` (candidate SKUs with their colourway variants, so
you can pick the right one) and `data/shot_plan.md` (which conditions to induce
per shoe, and why half the plan is deliberately single-condition).

Label into `manifest.csv` using the schema in `src/vpm/eval/manifest.py`. The
validator rejects unknown conditions, unknown item ids, duplicate photo ids and
conditions on clean controls; `check_split_disjoint` fails the run if any *item*
appears in both dev and test.

### Everything at once

```bash
./scripts/run_all.sh          # clean checkout -> reports/eval/report.md
pytest -q                     # 29 invariant tests, incl. the contamination guard
```

### Reproducing the backbone bake-off

```bash
python scripts/bakeoff.py --n-items 1200 --n-heldout 250 \
       --backbones siglip2-base dinov2-small-reg --whiten 0 256
```

---

## What each piece does

```
src/vpm/
  scrape/      sitemap harvesting, Myntra PDP parsing, resumable image download
  catalogue/   white-background trim, style clustering over the colourway graph
  embed/       backbone registry (SigLIP2 / DINOv2 ±registers / CLIP), saliency crop
  index/       exact flat index with view capping, PCA-whitening
  match/       query pipeline, open-set confidence features, conformal refusal
  eval/        Part B manifest schema, corruption operators, statistics, harness
```

**Design notes that are arguments, not plumbing** — each is defended in `REPORT.md`:

| Choice | Why |
|---|---|
| **SigLIP 2, not DINOv2** | Measured 0.672 vs 0.306 R@1 on this catalogue. My plan said DINOv2; the measurement overruled it. |
| **PCA-whitening** | Widens the same-item/different-item gap **15×**. Barely moves R@1. Separability and ranking are different axes. |
| **View capping at k=4** | Max-over-views inflates scores for items with more views (range here is 1–12), corrupting the refusal threshold. |
| **Style clusters** | 36% of naively held-out items kept a colourway twin in the index, making "absent" queries anything but. |
| **Conformal refusal** | The false-reject rate becomes a design parameter (α) rather than a tuned threshold. |
| **αQE / DBA default off** | Both inflate false accepts or destroy colourway distinctions. Built, measured, left off. |

---

## Data and provenance

Scraping is rate-limited (~10 req/s), identifies itself with a normal browser
user-agent, and records the source URL and fetch timestamp on every item.
`robots.txt` was checked before selecting the source: Myntra's is `Allow: /`
with disallows that do not cover product pages. Nike (`Disallow: */p/`), Adidas,
ASICS and New Balance were checked and rejected — the first by its rules, the
rest by anti-bot 403s.

`data/` and large artefacts are gitignored. The catalogue is reproducible from
step 1; nothing in the repo depends on a snapshot you cannot rebuild.

## Known limitations

Stated here rather than left to be discovered — the full accounting is in
`REPORT.md`:

1. **The 100 hand-shot photos are not in this repo.** Part B requires photographs
   of items the author physically owns. The manifest schema, validator,
   labelling template and harness are complete and tested; the synthetic stand-in
   exercises the identical code path. Every number produced from synthetic
   photos is labelled as such and must not be read as a Part B result.
2. **The bake-off protocol is pessimistic.** Holding out view 0 makes every query
   cross-view, which is harder than a real photo shot roughly side-on.
3. **The calibrator is fitted on synthetic corruptions** and would be deployed on
   real photos — a domain shift in the calibrator itself. The harness measures
   it (reliability diagram + ECE) rather than assuming it away.
4. **Catalogue snapshot drift.** A shoe bought two years ago may be delisted while
   a near-identical successor is listed. The manifest carries `sku_confidence`
   for exactly this, and metrics are reported with and without uncertain rows.
