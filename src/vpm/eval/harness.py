"""Run the matcher over a labelled test set and produce the graded tables.

Reporting choices that are arguments, not formatting:

* **Accuracy is reported at SKU and STYLE level.** Many SKUs differ only in a
  colourway panel, so at SKU level some photos pose an unanswerable question.
  Reporting only SKU accuracy takes a penalty for an impossible task; reporting
  only style accuracy hides the hard part.
* **The 3-way outcome table** separates accepted-and-wrong from refused. A bare
  FAR/FRR pair hides the wrong-accept rate, which is the error a user feels.
* **Per-condition cells carry intervals** and the write-up states that the
  ordering of conditions is not resolvable at n~100, rather than ranking noise.
* **Bootstrap resamples items**, because photos of one item are correlated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..match.pipeline import Matcher
from .manifest import CONDITIONS, Photo, condition_matrix
from .stats import (auroc, cluster_bootstrap, cooccurrence, dose_response,
                    marginal_effects, mcnemar_exact, risk_coverage, wilson)


@dataclass
class PhotoResult:
    photo_id: str
    item_id: int | None
    kind: str
    split: str
    conditions: list[str]
    top1: int | None
    top5: list[int]
    score: float
    correct_sku: bool
    correct_style: bool
    in_top5_sku: bool
    rank_of_truth: int | None     # None if truth not in the ranked list at all
    latency_ms: float
    p_match: float | None = None
    refused: bool | None = None
    set_size: int | None = None


class TestSetError(ValueError):
    pass


def check_items_indexed(photos: list[Photo], catalogue_ids: set[int]) -> None:
    """Every photo labelled in-catalogue must have its item IN the index.

    The brief requires Part B photos to be of items "genuinely in your
    catalogue". If an item is labelled in-catalogue but absent from the index,
    the photo is silently an out-of-catalogue query and every accuracy number
    computed from it is wrong -- accuracy is understated and the refusal AUROC
    collapses toward 0.5 because both classes are really the same class.

    This shipped broken once: the synthetic set was drawn from all 36,506 scraped
    items while the index held the first 3,000, so 34 of 40 "in-catalogue" items
    were never in the catalogue. Cheap to check, catastrophic to miss.
    """
    labelled = {p.item_id for p in photos
                if p.kind != "out_of_catalogue" and p.item_id is not None}
    missing = sorted(labelled - catalogue_ids)
    if missing:
        raise TestSetError(
            f"{len(missing)} of {len(labelled)} in-catalogue test items are NOT in the "
            f"index: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            "Every accuracy number would be wrong. Rebuild the index over these items, "
            "or regenerate the test set from the indexed slice.")


def run_photos(
    matcher: Matcher,
    photos: list[Photo],
    photo_root: Path,
    style_of: dict[int, int] | None = None,
    k: int = 5,
    full_rank: bool = True,
    calibrator=None,
    refuser=None,
) -> list[PhotoResult]:
    """Run the matcher over a labelled set.

    When a calibrator and conformal refuser are supplied, each photo also gets a
    calibrated `p_match` and an accept/refuse decision, which is what populates
    the three-way outcome table and the FAR/FRR figures.
    """
    from ..match.confidence import extract, image_quality

    style_of = style_of or {}
    sty = lambda p: style_of.get(int(p), int(p))
    out: list[PhotoResult] = []

    for ph in photos:
        path = Path(photo_root) / ph.filename
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        res = matcher.lookup(img, k=k)
        dt = (time.perf_counter() - t0) * 1000

        top5 = res.top_ids()
        top1 = top5[0] if top5 else None
        truth = ph.item_id

        rank = None
        if full_rank and truth is not None and res.scores_all is not None:
            # Median rank of the TRUTH even when wrong: "the right answer was
            # rank 3" and "rank 4700" are very different failures that top-5
            # accuracy scores identically.
            ids = matcher.index.unique_items
            pos = np.flatnonzero(ids == truth)
            if len(pos):
                rank = int((res.scores_all > res.scores_all[pos[0]]).sum()) + 1

        p_match = refused = set_size = None
        if calibrator is not None and res.scores_all is not None:
            feats = extract(res.scores_all, quality=image_quality(img),
                            salient_area=res.salient_area, final_top1=top1)
            p_match = float(calibrator.predict_proba(feats.values.reshape(1, -1))[0])
            if refuser is not None:
                ps = refuser.predict_set(top5, [c.score for c in res.candidates], p_match)
                refused, set_size = ps.refused, len(ps.items)

        out.append(PhotoResult(
            photo_id=ph.photo_id, item_id=truth, kind=ph.kind, split=ph.split,
            conditions=list(ph.conditions), top1=top1, top5=top5,
            score=float(res.candidates[0].score) if res.candidates else 0.0,
            correct_sku=bool(truth is not None and top1 == truth),
            correct_style=bool(truth is not None and top1 is not None and sty(top1) == sty(truth)),
            in_top5_sku=bool(truth is not None and truth in top5),
            rank_of_truth=rank, latency_ms=dt,
            p_match=p_match, refused=refused, set_size=set_size,
        ))
    return out


def _acc_block(rows: list[PhotoResult], field: str) -> dict:
    if not rows:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    y = np.array([getattr(r, field) for r in rows], float)
    g = np.array([r.item_id if r.item_id is not None else -1 for r in rows])
    ci = cluster_bootstrap(y, g)
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


def evaluate(results: list[PhotoResult], split: str = "test") -> dict:
    rows = [r for r in results if r.split == split]
    hard = [r for r in rows if r.kind == "hard"]
    clean = [r for r in rows if r.kind == "clean"]
    ooc = [r for r in rows if r.kind == "out_of_catalogue"]

    report: dict = {"split": split, "n_photos": len(rows),
                    "n_hard": len(hard), "n_clean": len(clean), "n_ooc": len(ooc)}

    for tag, block in (("hard", hard), ("clean", clean)):
        if not block:
            continue
        report[tag] = {
            "r1_sku": _acc_block(block, "correct_sku"),
            "r1_style": _acc_block(block, "correct_style"),
            "r5_sku": _acc_block(block, "in_top5_sku"),
            "median_rank_of_truth": float(np.median(
                [r.rank_of_truth for r in block if r.rank_of_truth is not None] or [float("nan")])),
            "latency_ms_median": float(np.median([r.latency_ms for r in block])),
        }

    # ---- the gap between the two halves, PAIRED by item ----
    if hard and clean:
        by_item_clean = {}
        for r in clean:
            by_item_clean.setdefault(r.item_id, []).append(r.correct_sku)
        pairs_c, pairs_h = [], []
        for r in hard:
            if r.item_id in by_item_clean:
                pairs_c.append(bool(np.mean(by_item_clean[r.item_id]) >= 0.5))
                pairs_h.append(r.correct_sku)
        if pairs_c:
            report["gap_paired"] = mcnemar_exact(np.array(pairs_c), np.array(pairs_h))
            report["gap_paired"]["n_pairs"] = len(pairs_c)

    # ---- per condition, with intervals ----
    per_cond = {}
    for c in CONDITIONS:
        block = [r for r in hard if c in r.conditions]
        if not block:
            continue
        k = sum(r.correct_sku for r in block)
        lo, hi = wilson(k, len(block))
        per_cond[c] = {
            "acc_sku": k / len(block), "wilson_lo": lo, "wilson_hi": hi, "n": len(block),
            "acc_style": float(np.mean([r.correct_style for r in block])),
            "boot": _acc_block(block, "correct_sku"),
        }
    report["per_condition"] = per_cond

    # ---- confounding + marginal effects ----
    if hard:
        fake = [Photo(r.photo_id, "", r.item_id, r.kind, r.split, r.conditions) for r in hard]
        M, names = condition_matrix(fake)
        co = cooccurrence(M, names)
        report["cooccurrence"] = {"counts": co["counts"], "per_photo": co["per_photo"],
                                  "names": co["names"],
                                  "jaccard": np.round(co["jaccard"], 3).tolist()}
        y = np.array([r.correct_sku for r in hard], int)
        g = np.array([r.item_id for r in hard])
        keep = [i for i, n in enumerate(names) if M[:, i].sum() >= 3]
        if keep and len(np.unique(y)) > 1:
            report["marginal_effects"] = marginal_effects(
                M[:, keep], y, [names[i] for i in keep], groups=g, n_boot=1000)
        report["dose_response"] = dose_response(M.sum(axis=1), y)

    # ---- refusal ----
    if ooc and (hard or clean):
        pos = np.array([r.p_match if r.p_match is not None else r.score for r in hard + clean])
        neg = np.array([r.p_match if r.p_match is not None else r.score for r in ooc])
        report["refusal"] = {"auroc": auroc(pos, neg), "n_pos": len(pos), "n_neg": len(neg)}
        if all(r.refused is not None for r in hard + clean + ooc):
            inc = hard + clean
            report["refusal"]["three_way"] = {
                "in_accepted_correct": int(sum(1 for r in inc if not r.refused and r.correct_sku)),
                "in_accepted_wrong": int(sum(1 for r in inc if not r.refused and not r.correct_sku)),
                "in_refused": int(sum(1 for r in inc if r.refused)),
                "ooc_accepted_FAR": float(np.mean([not r.refused for r in ooc])),
                "in_refused_FRR": float(np.mean([r.refused for r in inc])),
                "wrong_accept_rate": float(
                    np.mean([(not r.refused) and (not r.correct_sku) for r in inc])),
            }
        conf = np.array([r.p_match if r.p_match is not None else r.score for r in hard])
        report["risk_coverage"] = {
            k: v for k, v in risk_coverage(conf, np.array([r.correct_sku for r in hard])).items()
            if k != "curve"}
    return report


def to_markdown(report: dict) -> str:
    L: list[str] = []
    a = L.append
    a(f"## Results — split `{report['split']}`\n")
    a(f"{report['n_photos']} photos: {report['n_hard']} hard, {report['n_clean']} clean, "
      f"{report['n_ooc']} out-of-catalogue\n")

    a("### Headline\n")
    a("| set | R@1 SKU | R@1 style | R@5 SKU | median rank of truth | median latency |")
    a("|---|---|---|---|---|---|")
    for tag in ("clean", "hard"):
        if tag not in report:
            continue
        r = report[tag]
        f = lambda d: f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
        a(f"| {tag} | {f(r['r1_sku'])} | {f(r['r1_style'])} | {f(r['r5_sku'])} | "
          f"{r['median_rank_of_truth']:.0f} | {r['latency_ms_median']:.0f} ms |")
    a("")
    a("95% intervals are cluster bootstrap resampling **items**, not photos.\n")

    if "gap_paired" in report:
        g = report["gap_paired"]
        a("### The gap between the two halves (paired by item, McNemar exact)\n")
        a(f"- clean-right/hard-wrong: **{g['n01']}**, clean-wrong/hard-right: **{g['n10']}**")
        a(f"- accuracy drop: **{g['delta']*100:+.1f} pp** over {g['n_pairs']} pairs, p = {g['p_value']:.4g}\n")

    if report.get("per_condition"):
        a("### Accuracy by failure condition\n")
        a("| condition | n | R@1 SKU | 95% Wilson | R@1 style |")
        a("|---|---|---|---|---|")
        for c, d in sorted(report["per_condition"].items(), key=lambda kv: kv[1]["acc_sku"]):
            a(f"| {c} | {d['n']} | {d['acc_sku']:.3f} | "
              f"[{d['wilson_lo']:.3f}, {d['wilson_hi']:.3f}] | {d['acc_style']:.3f} |")
        a("")
        a("> Cells are small. At n=10 a 95% interval spans roughly ±25 pp, so the "
          "**ordering** of these conditions is not resolvable from this set. "
          "The marginal-effects table and the synthetic dose-response sweep are "
          "what the condition ranking should be read from.\n")

    if "marginal_effects" in report:
        a("### Marginal effect of each condition (exploratory)\n")
        a("| condition | n | log-odds | 95% CI | avg marginal effect |")
        a("|---|---|---|---|---|")
        for m in sorted(report["marginal_effects"], key=lambda d: d["marginal_pp"]):
            a(f"| {m['condition']} | {m['n']} | {m['log_odds']:+.2f} | "
              f"[{m['lo']:+.2f}, {m['hi']:+.2f}] | {m['marginal_pp']:+.1f} pp |")
        a("")
        a("> Conditions co-occur, so per-condition accuracy above is confounded. "
          "This fits correctness on the condition indicator matrix to estimate each "
          "condition's marginal contribution. Penalised and bootstrapped by item; "
          "**exploratory** at this sample size.\n")

    if "dose_response" in report:
        d = report["dose_response"]
        a("### Dose response\n")
        a(f"Each additional adverse condition multiplies the odds of a correct match by "
          f"**{d['odds_ratio']:.3f}**.\n")
        a("| # conditions | n | accuracy | 95% Wilson |")
        a("|---|---|---|---|")
        for lvl, v in sorted(d["per_level"].items()):
            a(f"| {lvl} | {v['n']} | {v['acc']:.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] |")
        a("")

    if "refusal" in report:
        r = report["refusal"]
        a("### Refusal\n")
        a(f"- AUROC in-catalogue vs out-of-catalogue: **{r['auroc']:.3f}** "
          f"({r['n_pos']} in, {r['n_neg']} out)")
        if "three_way" in r:
            t = r["three_way"]
            a("")
            a("| outcome | count / rate |")
            a("|---|---|")
            a(f"| in-catalogue, accepted & correct | {t['in_accepted_correct']} |")
            a(f"| in-catalogue, accepted & **wrong** | {t['in_accepted_wrong']} |")
            a(f"| in-catalogue, refused (FRR) | {t['in_refused']} ({t['in_refused_FRR']:.3f}) |")
            a(f"| out-of-catalogue, accepted (**FAR**) | {t['ooc_accepted_FAR']:.3f} |")
            a(f"| **wrong-accept rate** | {t['wrong_accept_rate']:.3f} |")
            a("")
            a("> The wrong-accept rate is the error a user actually feels, and a bare "
              "FAR/FRR pair hides it.\n")
    if "risk_coverage" in report:
        rc = report["risk_coverage"]
        a(f"- AURC **{rc['aurc']:.3f}**, excess over oracle (E-AURC) **{rc['e_aurc']:.3f}** "
          "— E-AURC isolates how well confidence *ranks* its own errors from how many there are.\n")
    return "\n".join(L)


def save(report: dict, results: list[PhotoResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (out_dir / "report.md").write_text(to_markdown(report))
    with (out_dir / "per_photo.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), default=float) + "\n")
