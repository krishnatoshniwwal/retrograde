"""
theme.py — Single source of truth for Retrograde's visual identity.

The palette is a pixel-art "night sky" scheme: deep navy skies, cream ink,
a lime primary, cyan highlights, a pale moon and sakura pink.  Every
renderer (Streamlit CSS, matplotlib figures, the PDF, the poster and the
Desmos export) pulls its colours from here so the whole app reads as one
system.

Importing this module also registers Retrograde's custom matplotlib
colormaps (``aurora``, ``sakura``, ``dusk``, ``moonlight``) so they can be
referenced by name anywhere a colormap string is accepted.
"""

from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

NIGHT = {
    "bg_deep":   "#061431",   # deepest sky / hard shadows
    "bg":        "#0b1f4b",   # page background
    "card":      "#112a60",   # raised surfaces
    "secondary": "#173571",   # hover / inset surfaces
    "border":    "#2b4c8a",   # pixel borders
    "fg":        "#f3f1e2",   # cream ink
    "muted":     "#a7b6d6",   # secondary text
    "primary":   "#a8e27c",   # lime — primary actions
    "accent":    "#6ad5e7",   # cyan — highlights
    "moon":      "#e3eda6",   # pale moon yellow
    "sakura":    "#e8a9c7",   # blossom pink
    "danger":    "#ff6568",
    "sky_1":     "#133d7e",
    "sky_2":     "#1c61a1",
    "sky_3":     "#2481aa",
}

PAPER = {
    "bg":      "#f4f0e1",   # cream paper
    "card":    "#ebe5d0",
    "field":   "#fbf8ee",
    "border":  "#cbbf9f",
    "fg":      "#0f2753",   # navy ink
    "muted":   "#4a5d80",
    "accent":  "#1c61a1",
    "sakura":  "#c76b98",
    "rule":    "#8a7f62",
}

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

CUSTOM_COLORMAPS: dict[str, list[str]] = {
    # sky blue → cyan → lime → moon
    "aurora":    ["#2481aa", "#6ad5e7", "#a8e27c", "#e3eda6"],
    # cyan → blossom → pale pink
    "sakura":    ["#6aa9c2", "#dca9c4", "#e8a9c7", "#f7dce8"],
    # dusk horizon: deep blue → pink → moon
    "dusk":      ["#1c61a1", "#86bccb", "#e8a9c7", "#e3eda6"],
    # cool moonlight: cyan → cream
    "moonlight": ["#5ea3be", "#a7b6d6", "#f3f1e2", "#e3eda6"],
}

#: Darker "ink" twins of the custom maps, for cream-paper output where the
#: pale night colours would wash out.
INK_COLORMAPS: dict[str, list[str]] = {
    "aurora_ink":    ["#0f2753", "#1c61a1", "#2481aa", "#2d8a4e"],
    "sakura_ink":    ["#1c61a1", "#6042a6", "#c76b98", "#d0587e"],
    "dusk_ink":      ["#0f2753", "#1c61a1", "#c76b98", "#a56a00"],
    "moonlight_ink": ["#0f2753", "#4a5d80", "#1c61a1", "#178f9c"],
}

#: Colormaps offered in the UI (custom first, then good matplotlib built-ins).
COLORMAP_CHOICES: list[str] = [
    "aurora", "sakura", "dusk", "moonlight",
    "plasma", "viridis", "cool", "spring", "turbo",
]


def register_colormaps() -> None:
    """Register custom colormaps with matplotlib (idempotent)."""
    for name, colors in {**CUSTOM_COLORMAPS, **INK_COLORMAPS}.items():
        if name in matplotlib.colormaps:
            continue
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list(name, colors, N=256)
        )


def paper_colormap(name: str) -> str:
    """Return the ink twin of a custom colormap for paper output (else ``name``)."""
    ink = f"{name}_ink"
    return ink if ink in INK_COLORMAPS else name


def colormap_hex_stops(name: str, n: int = 8) -> list[str]:
    """Sample ``n`` evenly spaced hex colours from a colormap.

    Used to build CSS gradient swatches and per-contour colours for the
    in-browser epicycle viewer.
    """
    register_colormaps()
    cmap = matplotlib.colormaps[name]
    out = []
    for i in range(n):
        r, g, b, _ = cmap(i / max(n - 1, 1))
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


register_colormaps()
