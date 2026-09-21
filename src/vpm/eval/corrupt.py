"""Parametrised photographic corruptions, one per failure condition.

Serves three consumers, which is why it is built carefully and once:
  1. the bake-off, as a stand-in for real query difficulty;
  2. the confidence calibrator, as its synthetic positive distribution;
  3. the automated stumper, as the search space it optimises over.

Severity runs 1..5 on every operator so dose-response curves are comparable
across conditions. The names match the Part B condition taxonomy exactly, so a
synthetic sweep and the hand-shot set can be plotted on the same axes.

Honest limitation, stated where it is implemented rather than buried in a
report: synthetic motion blur is a clean linear kernel, while real motion blur
is rolling-shutter-warped and coupled to exposure and sensor noise. These
operators are a proxy whose fidelity must itself be measured (see
`eval/harness.py`, synthetic-vs-real severity matching).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

CONDITIONS = [
    "low_light",
    "motion_blur",
    "defocus",
    "off_angle",
    "partial_occlusion",
    "cluttered_background",
    "specular_reflection",
    "small_in_frame",
    "jpeg_artifacts",
]


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def low_light(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    """Exposure loss plus the Poisson-Gaussian sensor noise that accompanies it.

    Gain is modelled too: real phones raise ISO in the dark, so noise and
    darkness are coupled rather than independent.
    """
    gain = [0.60, 0.42, 0.28, 0.18, 0.10][sev - 1]
    read = [2.0, 4.0, 7.0, 11.0, 16.0][sev - 1]
    a = _np(img).astype(np.float32) * gain
    a = rng.poisson(np.clip(a, 0, None)).astype(np.float32)
    a += rng.normal(0.0, read, a.shape)
    return _pil(a)


def motion_blur(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    k = [5, 11, 19, 29, 41][sev - 1]
    angle = float(rng.uniform(0, 180))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    s = kern.sum()
    kern = kern / s if s > 0 else np.ones((k, k), np.float32) / (k * k)
    return _pil(cv2.filter2D(_np(img), -1, kern))


def defocus(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    r = [2, 4, 7, 11, 15][sev - 1]
    k = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    kern = ((xx**2 + yy**2) <= r**2).astype(np.float32)
    kern /= kern.sum()
    return _pil(cv2.filter2D(_np(img), -1, kern))


def off_angle(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    """Perspective warp standing in for an oblique camera pose."""
    frac = [0.06, 0.12, 0.20, 0.30, 0.42][sev - 1]
    a = _np(img)
    h, w = a.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda: rng.uniform(-frac, frac)
    dst = np.float32(
        [
            [w * abs(jit()), h * abs(jit())],
            [w * (1 - abs(jit())), h * abs(jit())],
            [w * (1 - abs(jit())), h * (1 - abs(jit()))],
            [w * abs(jit()), h * (1 - abs(jit()))],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return _pil(cv2.warpPerspective(a, M, (w, h), borderValue=(128, 128, 128)))


def partial_occlusion(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    """Opaque patches covering a rising fraction of the frame.

    Area is only half the story -- occluding the logo hurts far more than
    occluding the same area of midsole -- which is why the hand-shot set carries
    a separate `occludes_logo` label that no area-based severity can capture.
    """
    frac = [0.05, 0.12, 0.22, 0.35, 0.50][sev - 1]
    a = _np(img).copy()
    h, w = a.shape[:2]
    remaining = frac * h * w
    while remaining > 0:
        bw = int(rng.uniform(0.15, 0.45) * w)
        bh = int(min(remaining / max(bw, 1), h * 0.6))
        if bw < 4 or bh < 4:
            break
        x = int(rng.uniform(0, max(1, w - bw)))
        y = int(rng.uniform(0, max(1, h - bh)))
        a[y : y + bh, x : x + bw] = rng.integers(0, 90, size=3)
        remaining -= bw * bh
    return _pil(a)


def cluttered_background(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    """Shrink the object into a busy synthetic scene.

    Scale and clutter move together because that is how the real failure
    presents: a photo taken from further away has both a smaller object and more
    background in frame.
    """
    fill = [0.75, 0.60, 0.45, 0.30, 0.18][sev - 1]
    a = _np(img)
    h, w = a.shape[:2]
    side = max(h, w)
    # Low-frequency coloured noise, upsampled: reads as an out-of-focus room.
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    bg = cv2.resize(small, (side, side), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), side * 0.02)
    nw, nh = int(w * fill), int(h * fill)
    obj = cv2.resize(a, (max(nw, 8), max(nh, 8)))
    x = int(rng.uniform(0, max(1, side - obj.shape[1])))
    y = int(rng.uniform(0, max(1, side - obj.shape[0])))
    bg[y : y + obj.shape[0], x : x + obj.shape[1]] = obj
    return _pil(bg)


def specular_reflection(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    alpha = [0.12, 0.25, 0.40, 0.55, 0.72][sev - 1]
    a = _np(img).astype(np.float32)
    h, w = a.shape[:2]
    glare = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rad = rng.uniform(0.15, 0.45) * max(h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        glare += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2)))
    glare = np.clip(glare, 0, 1)[..., None]
    return _pil(a * (1 - alpha * glare) + 255.0 * alpha * glare)


def small_in_frame(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    """Resolution loss: downscale then upscale back, destroying fine texture."""
    scale = [0.6, 0.42, 0.28, 0.18, 0.10][sev - 1]
    a = _np(img)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return _pil(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))


def jpeg_artifacts(img: Image.Image, sev: int, rng: np.random.Generator) -> Image.Image:
    q = [70, 50, 32, 20, 10][sev - 1]
    ok, buf = cv2.imencode(".jpg", _np(img)[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return _pil(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])


OPERATORS = {
    "low_light": low_light,
    "motion_blur": motion_blur,
    "defocus": defocus,
    "off_angle": off_angle,
    "partial_occlusion": partial_occlusion,
    "cluttered_background": cluttered_background,
    "specular_reflection": specular_reflection,
    "small_in_frame": small_in_frame,
    "jpeg_artifacts": jpeg_artifacts,
}


def apply(img: Image.Image, condition: str, severity: int, seed: int | None = None) -> Image.Image:
    if condition not in OPERATORS:
        raise KeyError(f"unknown condition {condition!r}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 1..5")
    rng = np.random.default_rng(seed)
    return OPERATORS[condition](img, severity, rng)


def apply_many(
    img: Image.Image, specs: list[tuple[str, int]], seed: int | None = None
) -> Image.Image:
    """Compose conditions, as real photos do (a dark photo is usually also blurry)."""
    rng = np.random.default_rng(seed)
    out = img
    for cond, sev in specs:
        out = OPERATORS[cond](out, sev, rng)
    return out
