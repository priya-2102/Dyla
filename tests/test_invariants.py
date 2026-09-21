"""Invariants that must hold, including the guards the write-up claims.

A mechanical guard is more convincing than a promise in a README, so the
contamination boundary and the view-count bias fix are asserted here rather than
merely described.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vpm.catalogue.trim import content_bbox, trim_to_square
from vpm.eval.corrupt import CONDITIONS, apply
from vpm.eval.manifest import ManifestError, Photo, check_split_disjoint
from vpm.eval.stats import auroc, cluster_bootstrap, mcnemar_exact, wilson
from vpm.index.flat import FlatIndex, cap_views
from vpm.index.pca import PCAWhitening
from vpm.match.confidence import FEATURE_NAMES, extract, feature_mask
from vpm.match.conformal import ConformalRefuser
from vpm.scrape.myntra import parse_product, resolve_image_url
from vpm.scrape.sitemap import parse_sitemap


# ---------- scraping ----------

def test_sitemap_parses_category_and_brand_from_url():
    xml = "<loc>https://www.myntra.com/Casual-Shoes/Big+Fox/slug-here/16427724/buy</loc>"
    refs = list(parse_sitemap(xml))
    assert len(refs) == 1
    assert refs[0].product_id == 16427724
    assert refs[0].category == "Casual-Shoes"
    assert refs[0].brand == "Big Fox"        # '+' decoded to a space


def test_sitemap_ignores_non_product_urls():
    assert list(parse_sitemap("<loc>https://www.myntra.com/shoes</loc>")) == []


def test_image_template_is_filled_and_forced_to_https():
    src = "http://assets.myntassets.com/h_($height),q_($qualityPercentage),w_($width)/v1/x.jpg"
    out = resolve_image_url(src, width=1080, height=1440, quality=90)
    assert out.startswith("https://")
    assert "h_1440,q_90,w_1080" in out
    assert "($" not in out


def test_parse_product_returns_none_on_junk():
    assert parse_product("<html>no myx here</html>") is None


# ---------- catalogue ----------

def test_trim_refuses_to_crop_a_blank_image():
    blank = Image.fromarray(np.full((200, 200, 3), 255, np.uint8))
    assert content_bbox(blank) is None
    out, trimmed = trim_to_square(blank, size=64)
    assert trimmed is False and out.size == (64, 64)


def test_trim_finds_the_object_and_squares_it():
    a = np.full((400, 300, 3), 245, np.uint8)
    a[100:300, 80:200] = [20, 40, 80]
    out, trimmed = trim_to_square(Image.fromarray(a), size=128)
    assert trimmed is True and out.size == (128, 128)


# ---------- index ----------

def test_cap_views_enforces_uniform_view_count():
    """The view-count bias fix: max-over-views inflates E[score] with view count."""
    rng = np.random.default_rng(0)
    emb, ids = [], []
    for item in range(20):
        base = rng.normal(size=32)
        base /= np.linalg.norm(base)
        for _ in range(int(rng.integers(1, 9))):     # 1..8 views, as in the real data
            v = base + 0.1 * rng.normal(size=32)
            emb.append(v / np.linalg.norm(v))
            ids.append(item)
    emb, ids = np.array(emb, np.float32), np.array(ids)
    keep = cap_views(emb, ids, k=4)
    counts = np.bincount(ids[keep])
    assert counts.max() <= 4


def test_index_returns_self_at_rank_one():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(60, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.arange(60)
    idx = FlatIndex(emb, ids)
    for probe in (0, 17, 59):
        assert idx.search(emb[probe], k=5).item_ids[0] == probe


def test_whitening_increases_separation():
    """The §5 claim, as a regression test."""
    rng = np.random.default_rng(2)
    shared = rng.normal(size=64) * 4.0          # a dominant shared direction
    A, B = [], []
    for _ in range(200):
        ident = rng.normal(size=64) * 0.3
        A.append(shared + ident + 0.05 * rng.normal(size=64))
        B.append(shared + ident + 0.05 * rng.normal(size=64))
    A = np.array(A, np.float32); B = np.array(B, np.float32)
    nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    An, Bn = nrm(A), nrm(B)
    raw_gap = (An * Bn).sum(1).mean() - (An @ Bn.T - np.eye(len(An)) * 9).max(1).mean()
    w = PCAWhitening(dim=32).fit(B)
    Aw, Bw = w.transform(A), w.transform(B)
    wht_gap = (Aw * Bw).sum(1).mean() - (Aw @ Bw.T - np.eye(len(Aw)) * 9).max(1).mean()
    assert wht_gap > raw_gap


# ---------- corruption ----------

@pytest.mark.parametrize("cond", CONDITIONS)
def test_every_corruption_runs_at_every_severity(cond):
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    for sev in range(1, 6):
        out = apply(img, cond, sev, seed=0)
        assert out.size[0] > 0 and out.mode == "RGB"


def test_corruption_rejects_bad_arguments():
    img = Image.fromarray(np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(KeyError):
        apply(img, "not_a_condition", 1)
    with pytest.raises(ValueError):
        apply(img, "low_light", 9)


# ---------- the contamination guard ----------

def test_dev_and_test_must_be_item_disjoint():
    photos = [
        Photo("a", "a.jpg", 1, "hard", "dev"),
        Photo("b", "b.jpg", 1, "hard", "test"),     # same ITEM in both splits
    ]
    with pytest.raises(ManifestError, match="item-disjoint"):
        check_split_disjoint(photos)


def test_item_disjoint_splits_pass():
    check_split_disjoint([
        Photo("a", "a.jpg", 1, "hard", "dev"),
        Photo("b", "b.jpg", 2, "hard", "test"),
    ])


# ---------- confidence + conformal ----------

def test_confidence_separates_a_match_from_an_absent_item():
    rng = np.random.default_rng(3)
    present = np.r_[0.95, rng.normal(0.30, 0.05, 499)]
    absent = np.r_[0.50, rng.normal(0.47, 0.05, 499)]
    f_present = extract(present).as_dict()
    f_absent = extract(absent).as_dict()
    assert f_present["margin_12"] > f_absent["margin_12"]
    assert f_present["zscore_q"] > f_absent["zscore_q"]


def test_quality_feature_mask_drops_the_right_columns():
    assert feature_mask(True).sum() == len(FEATURE_NAMES)
    assert feature_mask(False).sum() < len(FEATURE_NAMES)


def test_conformal_coverage_tracks_nominal():
    """FRR must be a design parameter, not an artefact."""
    rng = np.random.default_rng(4)
    calib = rng.beta(6, 2, 800)
    for alpha in (0.05, 0.1, 0.2):
        r = ConformalRefuser(alpha=alpha).fit(calib)
        fresh = rng.beta(6, 2, 2000)
        covered = np.mean([
            not r.predict_set([1], [1.0], s).refused for s in fresh
        ])
        assert abs(covered - (1 - alpha)) < 0.05


def test_conformal_needs_enough_calibration_points():
    with pytest.raises(ValueError):
        ConformalRefuser().fit(np.array([0.5, 0.6]))


# ---------- statistics ----------

def test_wilson_is_sane_at_the_extremes():
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and 0 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert 0.65 < lo < 1.0 and hi == pytest.approx(1.0, abs=1e-9)


def test_cluster_bootstrap_widens_with_clustering():
    """Correlated photos of one item must not be treated as independent.

    The realistic case: outcomes are correlated WITHIN an item -- some shoes the
    matcher gets right from every angle, others it never gets. Treating those 60
    photos as 60 independent samples understates the variance badly.
    """
    # 10 items x 6 photos; each item is entirely right or entirely wrong.
    y = np.repeat(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], float), 6)
    groups = np.repeat(np.arange(10), 6)
    independent = cluster_bootstrap(y, np.arange(len(y)), n_boot=2000)
    clustered = cluster_bootstrap(y, groups, n_boot=2000)
    assert (clustered.hi - clustered.lo) > (independent.hi - independent.lo)


def test_auroc_endpoints():
    assert auroc(np.array([3.0, 4, 5]), np.array([0.0, 1, 2])) == pytest.approx(1.0)
    assert auroc(np.array([0.0, 1, 2]), np.array([3.0, 4, 5])) == pytest.approx(0.0)


def test_mcnemar_is_symmetric_under_swap():
    a = np.array([1, 1, 1, 0, 0], bool)
    b = np.array([0, 0, 1, 0, 1], bool)
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(mcnemar_exact(b, a)["p_value"])
