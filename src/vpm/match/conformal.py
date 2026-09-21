"""Calibrated confidence and conformal refusal.

Two layers, deliberately separated:

`Calibrator` turns the ~18 open-set features into P(top-1 is correct AND the
item is in the catalogue). Logistic regression is chosen over a boosted tree not
because it scores better but because its coefficients are printable: "the
distractor-normalised margin carries more weight than raw top-1 similarity" is a
claim a reader can check, and a tree teaches them nothing.

`ConformalRefuser` converts that score into a prediction SET with a
distribution-free coverage guarantee: the true item is in the set with
probability >= 1-alpha. Refusal is then not a tuned threshold but the **empty
set**, and the false-reject rate is a design parameter rather than an artefact.
Set size doubles as a difficulty readout -- a singleton means confident, eight
means "it's one of these colourways", empty means not in the catalogue.

The guarantee holds under exchangeability between calibration and test. Ours is
fitted on synthetically corrupted catalogue images and deployed on real phone
photos, which is a domain shift IN THE CALIBRATOR ITSELF. So empirical coverage
is measured against nominal 1-alpha and the gap is reported: where the guarantee
breaks, the size of the break quantifies the shift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .confidence import FEATURE_NAMES, feature_mask


@dataclass
class Calibrator:
    include_quality: bool = True
    model: object | None = None
    iso: object | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    names_: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "Calibrator":
        """Fit on item-disjoint folds.

        `groups` must be item ids. Without item-disjoint folds the model
        memorises per-item score offsets and cross-validated AUROC is fiction.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        mask = feature_mask(self.include_quality)
        self.names_ = [n for n, m in zip(FEATURE_NAMES, mask) if m]
        Xm = np.asarray(X, float)[:, mask]
        self.mean_ = Xm.mean(axis=0)
        self.std_ = Xm.std(axis=0) + 1e-9
        Z = (Xm - self.mean_) / self.std_
        y = np.asarray(y, int)

        base = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
        n_groups = len(np.unique(groups)) if groups is not None else 0
        # Isotonic needs a lot of points; below that it overfits and Platt is safer.
        method = "isotonic" if len(y) >= 1000 else "sigmoid"
        if groups is not None and n_groups >= 5:
            # CalibratedClassifierCV does not forward `groups` to the splitter, so
            # materialise the item-disjoint folds as explicit index pairs. Folds
            # must be item-disjoint or the model memorises per-item score offsets.
            splitter = GroupKFold(n_splits=min(5, n_groups))
            folds = list(splitter.split(Z, y, groups=groups))
            folds = [(tr, te) for tr, te in folds if len(np.unique(y[tr])) > 1]
            cv = folds if folds else 3
        else:
            cv = 3
        self.model = CalibratedClassifierCV(base, method=method, cv=cv)
        self.model.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mask = feature_mask(self.include_quality)
        Z = (np.asarray(X, float)[:, mask] - self.mean_) / self.std_
        return self.model.predict_proba(Z)[:, 1]

    def coefficients(self) -> list[tuple[str, float]]:
        """Averaged logistic coefficients, largest magnitude first.

        This is the interpretability payoff and belongs in the write-up.
        """
        coefs = []
        for cc in getattr(self.model, "calibrated_classifiers_", []):
            est = getattr(cc, "estimator", None)
            if est is not None and hasattr(est, "coef_"):
                coefs.append(est.coef_[0])
        if not coefs:
            return []
        avg = np.mean(coefs, axis=0)
        return sorted(zip(self.names_, avg.tolist()), key=lambda t: -abs(t[1]))

    def save(self, path: Path) -> None:
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "Calibrator":
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)


@dataclass
class PredictionSet:
    items: list[int]
    p_values: list[float]
    refused: bool
    p_match: float

    def as_dict(self) -> dict:
        return {
            "items": self.items,
            "p_values": [round(p, 4) for p in self.p_values],
            "refused": self.refused,
            "p_match": round(self.p_match, 4),
        }


class ConformalRefuser:
    """Split conformal over candidate items.

    Calibration uses nonconformity scores of KNOWN-CORRECT matches. A candidate's
    conformal p-value is the fraction of calibration scores at least as
    nonconforming as it, so a candidate survives only if genuine matches are
    routinely no better than it. When no candidate survives, the set is empty and
    the system refuses.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.cal_scores_: np.ndarray | None = None

    def fit(self, calib_scores: np.ndarray) -> "ConformalRefuser":
        """`calib_scores` are p_match values for queries whose top-1 was CORRECT."""
        s = np.asarray(calib_scores, float)
        self.cal_scores_ = np.sort(s[np.isfinite(s)])
        if len(self.cal_scores_) < 20:
            raise ValueError("need >=20 calibration points for a usable conformal quantile")
        return self

    def p_value(self, score: float) -> float:
        """Conformal p-value for a candidate with confidence `score`.

        Nonconformity is -score, so the p-value is the fraction of calibration
        matches that were AT LEAST AS NONCONFORMING, i.e. scored no higher than
        this candidate. A high-scoring candidate sits above most genuine matches
        and gets a p-value near 1; a low-scoring one gets a p-value near 0 and
        falls out of the set.

        The +1 in numerator and denominator is the standard finite-sample
        correction that makes the coverage guarantee exact rather than asymptotic.
        """
        n = len(self.cal_scores_)
        n_at_least_as_bad = int(np.searchsorted(self.cal_scores_, score, side="left"))
        return (n_at_least_as_bad + 1) / (n + 1)

    def predict_set(
        self, candidate_items: list[int], candidate_scores: list[float], p_match: float
    ) -> PredictionSet:
        """Build the prediction set. An empty set IS the refusal."""
        # Candidates are ranked by similarity; the query-level p_match calibrates
        # the top-1, and lower-ranked candidates inherit a proportionally
        # discounted score so the set widens exactly when the top is not dominant.
        top = max(candidate_scores) if candidate_scores else 0.0
        keep_items, keep_p = [], []
        for it, sc in zip(candidate_items, candidate_scores):
            share = (sc / top) if top > 0 else 0.0
            pv = self.p_value(p_match * share)
            if pv > self.alpha:
                keep_items.append(int(it))
                keep_p.append(float(pv))
        return PredictionSet(keep_items, keep_p, refused=len(keep_items) == 0, p_match=float(p_match))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"alpha": self.alpha, "cal": self.cal_scores_.tolist()}))

    @staticmethod
    def load(path: Path) -> "ConformalRefuser":
        d = json.loads(Path(path).read_text())
        r = ConformalRefuser(alpha=d["alpha"])
        r.cal_scores_ = np.array(d["cal"], float)
        return r
