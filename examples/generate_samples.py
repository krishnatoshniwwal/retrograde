"""
examples/generate_samples.py — Generate synthetic sample images for testing.

Run from the project root:
    python examples/generate_samples.py
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


OUTPUT_DIR = Path(__file__).parent


def star_image(size: int = 400, n_points: int = 5) -> Image.Image:
    """Draw a star with inner concentric circles."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r_outer = size * 0.4
    r_inner = size * 0.17

    pts = []
    for i in range(n_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi / n_points * i - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, outline=(255, 255, 255))

    for r in [size * 0.1, size * 0.2, size * 0.3]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(160, 160, 160), width=2)

    return img


def circle_image(size: int = 300) -> Image.Image:
    """Simple circle — useful for verifying DFT accuracy."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    m = size // 10
    draw.ellipse([m, m, size - m, size - m], outline=(255, 255, 255), width=3)
    return img


def square_spiral_image(size: int = 400) -> Image.Image:
    """A square spiral for testing piecewise fits."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    step = size // 20
    x0, y0, x1, y1 = step, step, size - step, size - step
    while x0 < x1 and y0 < y1:
        draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))
        x0 += step * 2
        y0 += step * 2
        x1 -= step * 2
        y1 -= step * 2
    return img


def butterfly_image(size: int = 400) -> Image.Image:
    """Polar butterfly curve: r = e^(sin θ) - 2cos(4θ) + sin⁵((2θ-π)/24)"""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    scale = size * 0.18

    pts = []
    for i in range(2000):
        theta = 2 * math.pi * i / 2000 * 12  # 12 full rotations
        r = (
            math.exp(math.sin(theta))
            - 2 * math.cos(4 * theta)
            + math.sin((2 * theta - math.pi) / 24) ** 5
        )
        x = cx + scale * r * math.cos(theta)
        y = cy + scale * r * math.sin(theta)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(220, 180, 255), width=1)

    return img


def main():
    samples = {
        "star.png": star_image(400, 5),
        "circle.png": circle_image(300),
        "square_spiral.png": square_spiral_image(400),
        "butterfly.png": butterfly_image(400),
    }

    for fname, img in samples.items():
        out = OUTPUT_DIR / fname
        img.save(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
