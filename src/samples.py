"""
samples.py — Built-in demo shapes, drawn procedurally (no image files needed).

Every generator returns a PIL ``Image`` of white-ish strokes on black so the
Canny edge detector picks up clean outlines.  ``DEMO_SHAPES`` maps a display
name to its generator and is what the Streamlit demo picker uses.
"""

from __future__ import annotations

import math
from typing import Callable

from PIL import Image, ImageDraw


def _canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    return img, ImageDraw.Draw(img)


def _polyline(draw: ImageDraw.ImageDraw, pts, width: int = 3, closed: bool = True) -> None:
    if closed:
        pts = list(pts) + [pts[0]]
    draw.line(pts, fill=(255, 255, 255), width=width, joint="curve")


def star_image(size: int = 400, n_points: int = 5) -> Image.Image:
    """A star with three concentric rings."""
    img, draw = _canvas(size)
    cx = cy = size / 2
    r_outer, r_inner = size * 0.4, size * 0.17
    pts = []
    for i in range(n_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi / n_points * i - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, outline=(255, 255, 255))
    for r in (size * 0.1, size * 0.2, size * 0.3):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(180, 180, 180), width=2)
    return img


def retrograde_image(size: int = 400) -> Image.Image:
    """The app's namesake: the looping path of a planet seen from Earth.

    Geocentric position of an outer planet = (planet orbit) − (Earth orbit),
    which traces an epitrochoid with small retrograde loops.
    """
    img, draw = _canvas(size)
    c = size / 2
    pts = []
    n = 3000
    for i in range(n):
        t = 2 * math.pi * i / n
        # Earth: radius 1, 6 orbits; planet: radius 1.9, 1 orbit (ratio chosen to close)
        x = 1.9 * math.cos(t) - 1.0 * math.cos(6 * t)
        y = 1.9 * math.sin(t) - 1.0 * math.sin(6 * t)
        pts.append((c + x * size * 0.15, c + y * size * 0.15))
    _polyline(draw, pts, width=3)
    # The Sun at the centre
    r = size * 0.04
    draw.ellipse([c - r, c - r, c + r, c + r], outline=(255, 255, 255), width=3)
    return img


def heart_image(size: int = 400) -> Image.Image:
    """The classic parametric heart curve."""
    img, draw = _canvas(size)
    c = size / 2
    pts = []
    for i in range(1200):
        t = 2 * math.pi * i / 1200
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((c + x * size * 0.022, c - y * size * 0.022 - size * 0.03))
    _polyline(draw, pts, width=4)
    return img


def butterfly_image(size: int = 400) -> Image.Image:
    """Temple Fay's polar butterfly curve."""
    img, draw = _canvas(size)
    cx, cy = size / 2, size * 0.55
    scale = size * 0.16
    pts = []
    for i in range(4000):
        theta = 2 * math.pi * i / 4000 * 12
        r = (
            math.exp(math.sin(theta))
            - 2 * math.cos(4 * theta)
            + math.sin((2 * theta - math.pi) / 24) ** 5
        )
        pts.append((cx + scale * r * math.cos(theta), cy - scale * r * math.sin(theta)))
    _polyline(draw, pts, width=2, closed=False)
    return img


def rose_image(size: int = 400, k: int = 5) -> Image.Image:
    """A rhodonea (rose) curve with ``k`` petals inside a circle."""
    img, draw = _canvas(size)
    c = size / 2
    pts = []
    for i in range(2000):
        t = math.pi * i / 2000
        r = math.cos(k * t) * size * 0.38
        pts.append((c + r * math.cos(t), c + r * math.sin(t)))
    _polyline(draw, pts, width=3)
    r = size * 0.44
    draw.ellipse([c - r, c - r, c + r, c + r], outline=(200, 200, 200), width=2)
    return img


def circle_image(size: int = 300) -> Image.Image:
    """A single circle — the simplest possible Fourier series (one term)."""
    img, draw = _canvas(size)
    m = size // 10
    draw.ellipse([m, m, size - m, size - m], outline=(255, 255, 255), width=3)
    return img


def square_spiral_image(size: int = 400) -> Image.Image:
    """Nested squares — sharp corners stress-test the fit."""
    img, draw = _canvas(size)
    step = size // 20
    x0, y0, x1, y1 = step, step, size - step, size - step
    while x0 < x1 and y0 < y1:
        draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))
        x0 += step * 2
        y0 += step * 2
        x1 -= step * 2
        y1 -= step * 2
    return img


#: Display name -> generator, in the order shown in the demo picker.
DEMO_SHAPES: dict[str, Callable[[], Image.Image]] = {
    "Retrograde loop": retrograde_image,
    "Star":            star_image,
    "Heart":           heart_image,
    "Butterfly":       butterfly_image,
    "Rose":            rose_image,
    "Nested squares":  square_spiral_image,
}
