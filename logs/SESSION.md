# Session log

A factual record of how this system was built, in order, with the evidence that
moved each decision. Raw run logs are in `logs/runs/`.

**What this is honest about:** the build was AI-assisted (Claude Code). The log
below distinguishes three different things, because conflating them would
misrepresent how the work happened:

- **[H]** a decision I made, which the tool did not propose
- **[T→H]** the tool proposed something and I overruled it, with the evidence
- **[T]** the tool proposed something and I accepted it after checking

---

## Phase 0 — scoping

**[H] Category: footwear, not jewellery or eyewear.**
The tool offered jewellery/watches (closest to the stated domain, hardest CV
problem) and eyewear. I chose footwear on a logistics argument the tool had not
weighted: Part B needs 25–40 *distinct* items I can physically photograph **and**
that are currently listed. Footwear is the only category where I can reach that
count from my own household. Choosing the tractable domain silently looks like
luck; choosing it for a stated reason is scoping.

**[H] Scope policy: gate on quality.**
The tool's plan initially proposed committing to all four "take it further"
extensions. I rejected that: the brief explicitly says *"solve one properly
rather than four loosely"*, and grading rewards error analysis over feature
count. Ordered the extensions and gated each on the previous one being properly
analysed. Two extensions (multi-item, adaptive projection head) were later cut
under that gate — see `DECISIONS.md` D14.

**[H] Refusal set must be footwear that isn't listed, not mugs and keys.**
The tool offered "non-footwear objects" as an easier option. Rejected: the
nearest catalogue neighbour to a coffee mug is far away, so almost any threshold
refuses correctly and the resulting FAR/FRR is flattering and meaningless. The
honest case is a shoe the catalogue doesn't have.

---

## Phase 1 — source selection

**[T] Probe before committing.** Checked `robots.txt` and live pages for six
sources rather than assuming.

```
Nike        robots: "just crawl it" BUT Disallow: */p/  ; product-feed API 404
Adidas      403 (Akamai)
ASICS       403
New Balance error page
Puma IN     200, __NEXT_DATA__ + JSON-LD — works, but single-brand
Decathlon   9,148 products across ALL categories — footwear a fraction
Myntra      Allow: / ; 277 sitemaps x 30,000 URLs = 8.3M products
```

Selected Myntra. Sampled 120,000 URLs → footwear at **4.17%** → predicted ≈346k.
Full harvest returned **345,331**. The estimate held, which is a useful check
that the sampling wasn't biased by sitemap ordering.

---

## Phase 2 — the reversal that mattered most

**[T→H] The plan said DINOv2. The measurement said SigLIP 2.**

My written plan committed to DINOv2 ViT-B/14, with what looked like sound
reasoning: this is *instance*-level retrieval, and DINOv2's self-supervised
features are known to dominate instance retrieval while CLIP collapses distinct
items into a shared semantic bucket.

An architecture review I commissioned contradicted this, citing FORB (NeurIPS
2023) and ILIAS (CVPR 2025), which put CLIP/SigLIP ahead on **product**
retrieval. The reconciliation: DINOv2 wins on *landmarks* — rigid textured 3D
scenes under viewpoint change. Sneaker identity is carried by brand marks and
colourway.

I did not take either side on authority. I ran a bake-off on my own catalogue
(`scripts/bakeoff.py`, raw output in `logs/runs/bakeoff.json`):

```
backbone             dim   px  cleanR@1  cleanR@5  cleanAUC
dinov2-small-reg     768  224     0.303     0.479     0.455
dinov2-base         1536  224     0.233     0.393     0.457
siglip2-base        1536  256     0.672     0.878     0.656
```

SigLIP 2 by **2.2×**. My plan was wrong, and wrong for a nameable reason: I had
generalised a benchmark result across a domain boundary it does not cross.
DINOv2 ViT-B scoring *below* ViT-S was a second signal that these features are
simply mismatched to this data.

DINOv2 was not discarded — it is retained as the localiser, where dense spatial
structure is what's actually needed.

---

## Phase 3 — where I overruled the tool and was right

**[T→H] The review claimed CPU beats MPS at batch size 1.**

It recommended splitting device policy by call site: CPU for serving, MPS for
batch indexing. Its numbers:

```
ViT-S/14 @224 bs=1   CPU 19.0 ms   MPS 28.9 ms
ViT-B/14 @224 bs=1   CPU 69.0 ms   MPS 105.6 ms
```

I checked the provenance and found it had benchmarked **synthetic
shape-accurate transformer stacks**, not real timm graphs. I re-measured with
real weights, median of 30 runs after warm-up, with explicit `mps.synchronize()`:

```
dinov2-small   cpu  median= 21.0 ms   p10= 18.9  p90= 23.6
dinov2-small   mps  median= 11.2 ms   p10= 11.0  p90= 11.5
dinov2-base    cpu  median= 80.1 ms   p10= 78.7  p90= 85.2
dinov2-base    mps  median= 35.7 ms   p10= 31.8  p90= 38.7
```

MPS is **~2× faster**, not slower. Kept MPS for serving; discarded the
split-device recommendation.

Worth recording that the review's *downstream conclusion* survived its wrong
premise — it said the CPU-only 100k tier should use ViT-S, which is still right
because ViT-B costs 80 ms on CPU and leaves no headroom. A right answer from a
wrong measurement is still worth re-deriving.

---

## Phase 4 — three bugs found in my own evaluation

These were found before any headline number was written down. Each one produced
a number that was *impossible* rather than merely bad, which is what made them
findable.

### 4.1 R@1 = 0.001 — the corruption pipeline

First hard-condition run returned near-zero. Not a weak result, a broken one.

Cause: corruptions were applied to **224px thumbnails**. An 11-pixel motion-blur
kernel followed by a 0.45× shrink on an already-downsampled image leaves nothing.
Real adverse conditions are *optical* and precede the sensor's downsampling.

Fixed: corrupt at native resolution (1080px), then downsample.

### 4.2 AUROC = 0.455 — the hold-out wasn't open-set

Sub-chance AUROC, with held-out items scoring *higher* than in-catalogue ones.
Below chance is a bug, not a finding.

Diagnosed directly:

```
held-out items whose colourway variants are ALSO in the catalogue: 73/200 (36%)
```

Individually held-out items kept a near-identical twin in the index, so the
"absent" query was never absent. Fixed with union-find over the `colours` graph,
holding out whole style clusters. I would not have gone looking for that
metadata field without the broken number.

### 4.3 The saliency crop returned the full frame every time

Cause: DINOv2 without registers emits scattered high-norm artefact tokens in
background regions, so the bbox of all above-threshold patches spans everything.
Measured on a real catalogue image:

```
dinov2-small       components=4  largest=94/103  bbox rows 1-14   (inflated)
dinov2-small-reg   components=1  largest=103/103 bbox rows 3-12   (correct)
```

Two fixes, both kept: the register variant, and the bbox of the largest connected
component rather than of all above-threshold patches.

---

## Phase 5 — the unexpected result

Chasing the sub-chance AUROC produced the most useful finding in the project.
Measured same-item cross-view similarity against best-different-item similarity
over 250 items:

```
SAME item, view0 vs view1 : mean 0.9351
DIFF item, view0 vs best  : mean 0.9272
own other-view beats best OTHER item: 66.0%
```

Identity is worth **under 1%** of the similarity; the rest is viewpoint and
"this is a product photo of a shoe". The 66% figure exactly predicts the 0.672
R@1 — they are the same fact.

Whitening is the textbook fix. What was not expected is how unevenly it pays:

```
raw              same=0.9316 other=0.9263 gap=+0.0052 R@1=0.626
whitened d=128   same=0.5965 other=0.5480 gap=+0.0485 R@1=0.639
whitened d=256   same=0.5102 other=0.4360 gap=+0.0742 R@1=0.665
whitened d=512   same=0.4188 other=0.3381 gap=+0.0807 R@1=0.671
```

**15× on the separation gap, 4.5 pp on Recall@1.** Judged on ranking alone,
whitening looks like a rounding error you might drop. The refusal problem lives
entirely on the margin axis. Ranking quality and score-margin quality are
different properties of an embedding, and the hardest part of this brief depends
on the one Recall@1 cannot see.

---

## Phase 6 — engineering, two performance bugs of my own

**Index build ran at 5.5 views/s with the GPU 90% idle.** `ps` showed 9.8% CPU —
I/O and JPEG-decode bound, not compute bound. Added a prefetch thread pool.

**First fix was worse.** `ThreadPoolExecutor.map` submits every task immediately,
so all ~48k decoded images would be held in memory at once. Replaced with a
bounded sliding-window prefetch (depth = 4 × batch). CPU went 9.8% → 174% with
bounded memory.

---

## What the tool got right and I accepted after checking

- **View capping at k=4.** Max-over-views makes E[score] grow with view count,
  injecting a per-item prior into the refusal threshold. I checked the claim
  against the data before accepting: views per item range **1–12** (mean 6.08),
  so the bias was real and large.
- **αQE and DBA default off.** αQE drags queries toward the catalogue manifold,
  inflating top-1 for out-of-catalogue queries — it raises the exact rate the
  hardest extension is graded on. Built, kept behind flags, off.
- **Geometric verification as a confidence feature, not a re-ranker.** 40 inliers
  is near-proof; 2 inliers means "low texture", not "no match". One-sided
  evidence is useless for ranking and useful for calibration.
- **Bootstrap by item, not by photo.** Verified with a regression test that the
  interval actually widens under clustering.

---

## Open, and honestly so

The 100 hand-shot photographs are not in this repo. Part B requires photographs
of items I physically own, and I will not present synthetic images as hand-shot
ones. The schema, validator, contamination guard, corruption operators,
statistics and harness are complete and tested, and the synthetic stand-in
exercises the identical code path — but that work is not done until the camera
comes out.
