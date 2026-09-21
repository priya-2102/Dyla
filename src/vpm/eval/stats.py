"""Statistics for an evaluation with ~100 photos and co-occurring labels.

Two problems drive every choice here:

1. **Tiny cells.** Nine conditions over ~120 photos leaves n=8-15 per condition.
   8/12 = 67% has a 95% interval of roughly [39%, 86%], so any *ordering* of
   conditions read off raw cell accuracies is noise. Every per-condition number
   ships with an interval, and the write-up says the ordering is unresolvable at
   this n rather than pretending otherwise.

2. **Conditions co-occur, causally.** Low light forces a longer exposure, which
   causes motion blur. Naive per-condition accuracy therefore charges the joint
   damage to each condition separately, and whichever condition most often
   accompanies the genuinely destructive one gets blamed for its harm.

Photos of one item are also correlated -- same object, same lighting, same
session -- so resampling photos i.i.d. understates variance. The bootstrap
resamples ITEMS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] n={self.n}"


def cluster_bootstrap(
    correct: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile CI for a mean, resampling clusters (items) with replacement."""
    correct = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    if len(correct) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    uniq, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_group), len(by_group))
        idx = np.concatenate([by_group[i] for i in pick])
        stats[b] = correct[idx].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(correct.mean()), float(lo), float(hi), int(len(correct)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- sane at the 0/n and n/n extremes where Wald is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float((c - half) / d), float((c + half) / d))


def mcnemar_exact(clean_correct: np.ndarray, hard_correct: np.ndarray) -> dict:
    """Exact McNemar on PAIRED clean/hard outcomes for the same items.

    Pairing removes item difficulty as a confounder, which matters because the
    photographer's choice of which items to shoot under which conditions is not
    random. Far better powered than comparing two independent groups at n~40.
    """
    from math import comb

    a = np.asarray(clean_correct, dtype=bool)
    b = np.asarray(hard_correct, dtype=bool)
    n01 = int(np.sum(a & ~b))   # clean right, hard wrong
    n10 = int(np.sum(~a & b))   # clean wrong, hard right
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "delta": 0.0}
    # two-sided exact binomial test at p=0.5
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return {
        "n01": n01,
        "n10": n10,
        "p_value": float(min(1.0, 2 * tail)),
        "delta": float(a.mean() - b.mean()),
    }


def cooccurrence(cond_matrix: np.ndarray, names: list[str]) -> dict:
    """Jaccard overlap between conditions, plus how many co-occur per photo.

    This is the FIRST table in the evaluation section, because it tells the
    reader how much to trust every per-condition number that follows.
    """
    M = np.asarray(cond_matrix, dtype=bool)
    k = M.shape[1]
    jac = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            inter = np.sum(M[:, i] & M[:, j])
            union = np.sum(M[:, i] | M[:, j])
            jac[i, j] = inter / union if union else 0.0
    return {
        "names": names,
        "jaccard": jac,
        "counts": M.sum(axis=0).tolist(),
        "per_photo": np.bincount(M.sum(axis=1), minlength=1).tolist(),
    }


def marginal_effects(
    cond_matrix: np.ndarray,
    correct: np.ndarray,
    names: list[str],
    l2: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> list[dict]:
    """L2-penalised logistic regression of correctness on condition indicators.

    Reported as EXPLORATORY. With ~9 correlated predictors on ~120 points this
    estimates rather than confirms, and small cells produce separation that would
    otherwise send coefficients to infinity -- hence the penalty, and hence
    bootstrap intervals instead of asymptotic standard errors.

    Bootstrap resamples items, matching the clustering in the accuracy CIs.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(cond_matrix, dtype=float)
    y = np.asarray(correct, dtype=int)
    if len(np.unique(y)) < 2:
        return [{"condition": n, "log_odds": float("nan"), "lo": float("nan"),
                 "hi": float("nan"), "marginal_pp": float("nan")} for n in names]

    def fit(Xs, ys):
        # sklearn >=1.8 deprecated the `penalty` kwarg; L2 is the default and C
        # carries the strength.
        m = LogisticRegression(C=1.0 / max(l2, 1e-9), max_iter=2000)
        m.fit(Xs, ys)
        return m

    base = fit(X, y)
    coefs = base.coef_[0]

    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)

    boot = np.full((n_boot, X.shape[1]), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_group), len(by_group))
        idx = np.concatenate([by_group[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boot[b] = fit(X[idx], y[idx]).coef_[0]
        except Exception:
            continue

    out = []
    for i, name in enumerate(names):
        col = boot[:, i]
        col = col[~np.isnan(col)]
        lo, hi = (np.percentile(col, [2.5, 97.5]) if len(col) > 20 else (np.nan, np.nan))
        # Average marginal effect: flip this condition on for every photo and
        # average the change in predicted probability. Log-odds are not readable.
        X1, X0 = X.copy(), X.copy()
        X1[:, i], X0[:, i] = 1.0, 0.0
        amp = float((base.predict_proba(X1)[:, 1] - base.predict_proba(X0)[:, 1]).mean() * 100)
        out.append({
            "condition": name,
            "log_odds": float(coefs[i]),
            "lo": float(lo),
            "hi": float(hi),
            "marginal_pp": amp,
            "n": int(X[:, i].sum()),
        })
    return out


def dose_response(n_conditions: np.ndarray, correct: np.ndarray) -> dict:
    """Accuracy vs how many adverse conditions are present.

    One coefficient instead of nine, so it is far better powered -- "each extra
    adverse condition multiplies the odds of a correct match by ~x" is defensible
    where nine separate coefficients are not.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(n_conditions, dtype=float).reshape(-1, 1)
    y = np.asarray(correct, dtype=int)
    if len(np.unique(y)) < 2:
        return {"odds_ratio": float("nan"), "per_level": {}}
    m = LogisticRegression(max_iter=2000).fit(x, y)
    per = {}
    for lvl in sorted(set(x.ravel().astype(int))):
        sel = x.ravel() == lvl
        k, n = int(y[sel].sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        per[int(lvl)] = {"acc": k / n if n else float("nan"), "lo": lo, "hi": hi, "n": n}
    return {"odds_ratio": float(np.exp(m.coef_[0][0])), "per_level": per}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-safe."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                      # average ranks within ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Risk-coverage curve and AURC, plus the oracle for E-AURC.

    E-AURC (excess over the oracle ordering) isolates how well the confidence
    score RANKS its own errors from how many errors there are, which is the only
    fair way to compare two confidence scorers at different accuracies.
    """
    conf = np.asarray(confidence, float)
    corr = np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return {"aurc": float("nan"), "e_aurc": float("nan"), "curve": []}
    order = np.argsort(-conf)
    err = 1.0 - corr[order]
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    # Oracle: every correct answer ranked above every incorrect one.
    oracle_err = 1.0 - np.sort(corr)[::-1]
    oracle_risk = np.cumsum(oracle_err) / np.arange(1, n + 1)
    return {
        "aurc": aurc,
        "e_aurc": float(aurc - np.mean(oracle_risk)),
        "curve": [{"coverage": float(c), "risk": float(r)} for c, r in zip(cov, risk)],
    }
