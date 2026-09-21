"""Part B test-set schema, validation, and the contamination guard.

The schema is deliberately strict and fails loudly. A labelling flaw discovered
after 130 photos have been shot is unrecoverable without reshooting, so unknown
conditions, unknown item ids and duplicate photo ids are errors, not warnings.

Splits are fixed at labelling time, before any model has seen a photo:

  dev     threshold selection and sanity checks   (~30 in + ~10 out)
  test    touched exactly once, at the end        (100 in + 20 out)

They must be disjoint in ITEMS, not merely in photos: two photos of the same
shoe share an object, a lighting rig and a session, so an item in both splits
lets the calibrator learn item-specific score offsets and makes the test number
a fantasy.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

CONDITIONS = [
    "low_light", "harsh_flash", "backlit", "motion_blur", "defocus",
    "off_angle", "partial_occlusion", "hand_or_foot_in_frame",
    "cluttered_background", "specular_reflection", "small_in_frame",
    "cropped_item", "worn_dirty",
]

SPLITS = {"dev", "test"}
KINDS = {"hard", "clean", "out_of_catalogue"}

# How confident we are that the photographed SKU is the labelled catalogue SKU.
# `style_match` means the model matched but the exact colourway did not -- those
# rows are genuinely out-of-catalogue at SKU level and are the hardest possible
# refusal negatives. Metrics are reported with and without `uncertain`.
SKU_CONFIDENCE = {"exact", "style_match", "uncertain"}

HEADER = [
    "photo_id", "filename", "item_id", "kind", "split",
    "conditions", "occludes_logo", "sku_confidence", "notes",
]


@dataclass
class Photo:
    photo_id: str
    filename: str
    item_id: int | None          # None for out_of_catalogue
    kind: str
    split: str
    conditions: list[str] = field(default_factory=list)
    occludes_logo: bool = False
    sku_confidence: str = "exact"
    notes: str = ""

    def n_conditions(self) -> int:
        return len(self.conditions)


class ManifestError(ValueError):
    pass


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_manifest(
    path: Path, catalogue_ids: set[int] | None = None, photo_root: Path | None = None
) -> list[Photo]:
    rows: list[Photo] = []
    seen: set[str] = set()
    errors: list[str] = []

    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(HEADER) - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"manifest is missing columns: {sorted(missing)}")

        for i, row in enumerate(reader, 2):   # row 1 is the header
            pid = (row.get("photo_id") or "").strip()
            if not pid:
                errors.append(f"row {i}: empty photo_id")
                continue
            if pid in seen:
                errors.append(f"row {i}: duplicate photo_id {pid!r}")
            seen.add(pid)

            kind = (row.get("kind") or "").strip()
            if kind not in KINDS:
                errors.append(f"row {i}: kind {kind!r} not in {sorted(KINDS)}")

            split = (row.get("split") or "").strip()
            if split not in SPLITS:
                errors.append(f"row {i}: split {split!r} not in {sorted(SPLITS)}")

            raw_item = (row.get("item_id") or "").strip()
            item_id: int | None = None
            if kind == "out_of_catalogue":
                if raw_item:
                    errors.append(f"row {i}: out_of_catalogue rows must have an empty item_id")
            else:
                if not raw_item.isdigit():
                    errors.append(f"row {i}: item_id {raw_item!r} is not an integer")
                else:
                    item_id = int(raw_item)
                    if catalogue_ids is not None and item_id not in catalogue_ids:
                        errors.append(f"row {i}: item_id {item_id} is not in the catalogue")

            conds = [c.strip() for c in (row.get("conditions") or "").split("|") if c.strip()]
            for c in conds:
                if c not in CONDITIONS:
                    errors.append(f"row {i}: unknown condition {c!r}")
            if kind == "clean" and conds:
                errors.append(f"row {i}: clean control photos must have no conditions")

            sku_conf = (row.get("sku_confidence") or "exact").strip()
            if sku_conf not in SKU_CONFIDENCE:
                errors.append(f"row {i}: sku_confidence {sku_conf!r} not in {sorted(SKU_CONFIDENCE)}")

            fn = (row.get("filename") or "").strip()
            if photo_root is not None and fn and not (Path(photo_root) / fn).exists():
                errors.append(f"row {i}: file not found: {fn}")

            rows.append(Photo(pid, fn, item_id, kind, split, conds,
                              _parse_bool(row.get("occludes_logo", "")), sku_conf,
                              (row.get("notes") or "").strip()))

    if errors:
        raise ManifestError(
            f"{len(errors)} problem(s) in {path}:\n  " + "\n  ".join(errors[:40])
        )
    return rows


def check_split_disjoint(photos: list[Photo]) -> None:
    """Fail if any ITEM appears in both dev and test. A mechanical guard beats a promise."""
    dev = {p.item_id for p in photos if p.split == "dev" and p.item_id is not None}
    test = {p.item_id for p in photos if p.split == "test" and p.item_id is not None}
    overlap = dev & test
    if overlap:
        raise ManifestError(
            f"{len(overlap)} item(s) appear in BOTH dev and test: {sorted(overlap)[:10]} ... "
            "splits must be item-disjoint, not merely photo-disjoint"
        )


def condition_matrix(photos: list[Photo]) -> tuple["object", list[str]]:
    import numpy as np
    M = np.zeros((len(photos), len(CONDITIONS)), dtype=float)
    for i, p in enumerate(photos):
        for c in p.conditions:
            M[i, CONDITIONS.index(c)] = 1.0
    return M, list(CONDITIONS)


def write_template(path: Path, n: int = 0) -> None:
    """Emit an empty manifest with the header and one commented example row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerow([
            "p0001", "hard/IMG_0001.jpg", "16427724", "hard", "test",
            "low_light|motion_blur", "false", "exact", "shot indoors at dusk",
        ])
    print(f"wrote template -> {path}")
    print(f"conditions vocabulary: {', '.join(CONDITIONS)}")


def summarise(photos: list[Photo]) -> dict:
    from collections import Counter
    by_kind = Counter(p.kind for p in photos)
    by_split = Counter(p.split for p in photos)
    conds = Counter(c for p in photos for c in p.conditions)
    items = {p.item_id for p in photos if p.item_id is not None}
    return {
        "photos": len(photos),
        "items": len(items),
        "by_kind": dict(by_kind),
        "by_split": dict(by_split),
        "conditions": dict(conds),
        "mean_conditions_per_hard_photo": (
            sum(p.n_conditions() for p in photos if p.kind == "hard")
            / max(sum(1 for p in photos if p.kind == "hard"), 1)
        ),
    }
