# Decision log

Chronological. Each entry records what was decided, what evidence moved it, and —
where it applies — what it overruled. Entries marked **⟲ reversal** are places
where I changed my own mind against a written plan, and **⚡ overrule** where I
went against the AI review I had commissioned.

---

### D1 — Category: footwear, not jewellery
Jewellery and watches are closest to the stated domain and are the harder CV
problem (low texture, specular highlights dominate appearance). I chose footwear
anyway, deliberately, for a reason I can defend: Part B requires 25–40 *distinct*
items I can physically photograph and that are *currently listed*. Footwear is
the only category where I can reach that count. Choosing the tractable domain
silently looks like luck; choosing it with a stated reason is a scoping decision.

### D2 — Source: Myntra, after rejecting four alternatives
Probed `robots.txt` and live pages for Nike, Adidas, ASICS, New Balance, Puma and
Decathlon before writing a scraper. Nike disallows `*/p/`; three return 403;
Puma is single-brand; Decathlon has 9,148 products across *all* categories.
Myntra: 277 sitemaps × 30k URLs, product pages allowed, and **category + brand in
the URL path** so footwear is filterable with zero page fetches.
Estimated footwear at 4.17% of the corpus from a 120k-URL sample → ≈346k.
Full harvest: **345,331**. Estimate held.

### D3 — Coverage probe before bulk scraping
The failure mode that sinks this assignment is scraping 5,000 images, shooting
120 photos, then finding a third of the SKUs absent. So the inventory of
photographable pairs resolves to product ids *before* the camera comes out, and
items whose exact colourway is delisted get labelled `model_present_colourway_absent`
— which makes them the hardest possible refusal negatives rather than lost data.
**Still outstanding: this needs the physical inventory.**

### D4 — ⟲ reversal: SigLIP 2 over DINOv2
My plan committed to DINOv2 on instance-retrieval grounds. That reasoning holds
for *landmarks*; FORB and ILIAS both put CLIP/SigLIP ahead on *products*. Rather
than argue from literature I ran a bake-off on my own catalogue:
**SigLIP 2 0.672 vs DINOv2 0.306 / 0.233 clean R@1.** 2.2×. Plan overruled.
DINOv2 retained as the *localiser*, where dense spatial structure is what's needed.

### D5 — ⚡ overrule: MPS, not CPU, for serving
The review reported CPU faster than MPS at batch size 1 and recommended splitting
device policy by call site. It had benchmarked *synthetic* transformer stacks.
Re-measured with real timm weights: MPS **~2× faster** (ViT-S 11.2 vs 21.0 ms;
ViT-B 35.7 vs 80.1 ms). Kept MPS. Its *conclusion* for the CPU-only 100k tier
(use ViT-S) survived the wrong premise, so that part was adopted.

### D6 — View capping at k=4
Accepted from the review, and it proved more necessary than argued. Max-over-views
makes E[score] grow with view count, injecting a per-item prior into the exact
quantity the refusal threshold reads. Measured in this catalogue: **views per item
range 1–12** (mean 6.08). Greedy max-min diverse selection to a uniform k.

### D7 — Style clusters, forced by a sub-chance AUROC
First open-set measurement returned **AUROC 0.455** — worse than random. A metric
below chance is a bug. Cause: **36% of held-out items still had a colourway twin
in the index**. Fixed with union-find over the `colours` graph, holding out whole
clusters. This is why that metadata field is load-bearing, and I would not have
gone looking for it without the broken number.

### D8 — Corrupt at native resolution
First hard-condition run: **R@1 = 0.001**. Cause: corrupting 224px thumbnails, so
an 11px blur kernel plus a 0.45× shrink left nothing. Real adverse conditions are
optical and precede downsampling. Now corrupts at 1080px, then downsamples.

### D9 — Registers + largest connected component for saliency
The patch-norm localiser returned the full frame on every image. DINOv2 without
registers emits scattered high-norm artefact tokens in background regions.
Measured: plain ViT-S gave **4 components** (9 stray patches, bbox rows 1–14);
the `reg4` variant gave **1** (rows 3–12). Both fixes kept.

### D10 — Whitening, kept for a reason R@1 would have hidden
Measured the cone effect directly: same-item cross-view cosine **0.9351** vs
best-different-item **0.9272** — identity is worth **0.8%** of similarity.
Whitening widens that gap **15×** (0.0052 → 0.0807) but moves R@1 only **+4.5 pp**.
Judged on ranking alone, whitening looks droppable. The refusal problem lives
entirely on the margin axis, so it is essential there. **Ranking quality and
score-margin quality are different properties.**

### D11 — Conformal refusal instead of a tuned threshold
Verified coverage tracks nominal within 2 pp at α ∈ {0.05, 0.10, 0.20}, with
**FRR ≈ α by construction**. Turns one of the two graded rates into a design
parameter. Precondition (exchangeability) is knowingly violated by the synthetic→
real shift; the harness measures the violation rather than assuming it away.

### D12 — αQE and DBA built, measured, defaulted off
αQE drags the query toward the catalogue manifold, inflating top-1 for
out-of-catalogue queries and raising the false-accept rate the hardest extension
is graded on. DBA averages colourway clusters toward their centroid, destroying
the distinction that matters most here. Both behind flags, off.

### D13 — Geometric verification demoted to a feature
40 RANSAC inliers is near-proof of a match; 2 inliers means "low texture", not
"no match". One-sided evidence: useless for ordering the low-texture majority,
useful as one calibrator input among many.

### D14 — Scope cut: multi-item and the adaptive head
Both planned, both cut. The brief says solve one thing properly rather than four
loosely, and rewards error analysis over feature count. Cutting them is the
choice that keeps §4 and §5 of the report deep.

### D15 — No synthetic photos presented as hand-shot
The synthetic test set exists so the harness runs from a clean checkout and as
the automated stumper. It is labelled as such everywhere and its numbers are
never reported as Part B results.
