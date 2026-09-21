"""Canonicalise framing so catalogue and query land on the same canvas.

The domain gap is mostly a *framing* gap: catalogue shots are a tight object on
white, real photos are an object occupying a fraction of a cluttered frame. The
embedding difference that follows is a nuisance variable we control on both
sides. Trimming the catalogue to the object's bounding box and padding to square
removes scale and centering as free variables before any model sees the image.

Deliberately dependency-free (Pillow + numpy) and ~2 ms/image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Pad colour. Grey rather than white: a white sneaker on a white canvas has no
# boundary, and both DINOv2 and SigLIP shift behaviour on pure-white product
# backdrops in a way a segmented query photo will not share.
PAD_RGB = (128, 128, 128)


def content_bbox(img: Image.Image, tol: int = 8, min_frac: float = 0.05) -> tuple[int, int, int, int] | None:
    """Bounding box of non-background content, or None if detection is unsafe.

    Background is inferred from the image's own border pixels rather than
    assumed to be white, so this also handles the light-grey and off-white
    backdrops that appear across the catalogue.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    h, w, _ = a.shape
    if h < 8 or w < 8:
        return None

    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)

    # A pixel is "content" if it differs from the inferred background.
    diff = np.abs(a - bg).max(axis=2)
    mask = diff > tol

    if not mask.any():
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    # Refuse implausible crops: a near-empty box means the background guess was
    # wrong, and a near-full box means there was nothing to trim. In both cases
    # returning None and using the original frame is the safe outcome.
    area_frac = ((bottom - top) * (right - left)) / float(h * w)
    if area_frac < min_frac or area_frac > 0.995:
        return None
    return left, top, right, bottom


def trim_to_square(
    img: Image.Image,
    margin: float = 0.06,
    size: int | None = None,
    pad_rgb: tuple[int, int, int] = PAD_RGB,
) -> tuple[Image.Image, bool]:
    """Crop to content with a relative margin and pad to a square canvas.

    Returns (image, trimmed) so callers can record how often trimming actually
    fired -- a segmentation/trim failure is a cascading error worth measuring
    separately from retrieval error.
    """
    img = img.convert("RGB")
    box = content_bbox(img)
    trimmed = box is not None
    if trimmed:
        left, top, right, bottom = box
        bw, bh = right - left, bottom - top
        mx, my = int(bw * margin), int(bh * margin)
        left, top = max(0, left - mx), max(0, top - my)
        right, bottom = min(img.width, right + mx), min(img.height, bottom + my)
        img = img.crop((left, top, right, bottom))

    side = max(img.width, img.height)
    canvas = Image.new("RGB", (side, side), pad_rgb)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    if size is not None:
        canvas = canvas.resize((size, size), Image.BICUBIC)
    return canvas, trimmed
