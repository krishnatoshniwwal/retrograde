"""Tests for the theme, demo shapes and UI HTML builders."""

import json
import re

import matplotlib
import numpy as np
import pytest

from src.contours import get_all_contours
from src.fourier import dft_pipeline
from src.preprocessing import preprocess_array
from src.render import get_theme, plot_poster, fig_to_png_bytes
from src.samples import DEMO_SHAPES
from src.theme import (
    COLORMAP_CHOICES, CUSTOM_COLORMAPS, INK_COLORMAPS, colormap_hex_stops, paper_colormap,
)
from src import ui


# ── theme ────────────────────────────────────────────────────────────────

def test_custom_colormaps_registered():
    for name in list(CUSTOM_COLORMAPS) + list(INK_COLORMAPS):
        assert name in matplotlib.colormaps


def test_all_ui_colormaps_resolve():
    for name in COLORMAP_CHOICES:
        assert name in matplotlib.colormaps


def test_hex_stops_format():
    stops = colormap_hex_stops("aurora", 6)
    assert len(stops) == 6
    assert all(re.fullmatch(r"#[0-9a-f]{6}", s) for s in stops)


def test_paper_colormap_mapping():
    assert paper_colormap("aurora") == "aurora_ink"
    assert paper_colormap("viridis") == "viridis"


def test_theme_aliases():
    assert get_theme("dark") is get_theme("night")
    assert get_theme("light") is get_theme("paper")
    assert get_theme("night")["bg"] != get_theme("paper")["bg"]


# ── samples ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(DEMO_SHAPES))
def test_demo_shapes_yield_contours(name):
    img = np.array(DEMO_SHAPES[name]().convert("RGB"))
    edges, _ = preprocess_array(img)
    contours = get_all_contours(edges, simplify=True)
    assert len(contours) >= 1


# ── ui builders ──────────────────────────────────────────────────────────

def _fourier_results(n=3):
    img = np.array(DEMO_SHAPES["Star"]().convert("RGB"))
    edges, _ = preprocess_array(img)
    contours = get_all_contours(edges, simplify=True)[:n]
    return [dft_pipeline(c, {"fourier": {"n_terms": 30}}) for c in contours]


def test_hero_has_no_blank_lines():
    html = ui.hero()
    assert "\n" not in html  # Markdown would end the raw-HTML block at a blank line
    assert "Retrograde" in html and "rg-spin" in html


def test_epicycle_machine_nesting_balanced():
    svg = ui.epicycle_machine_svg()
    assert svg.count("<g") == svg.count("</g>")


def test_viewer_embeds_valid_json():
    results = _fourier_results()
    html = ui.epicycle_viewer_html(results, [r["path"] for r in results], n_terms=30,
                                   colormap="aurora")
    assert not re.search(r"__[A-Z0-9_]+__", html)  # all template placeholders filled
    m = re.search(r"var D = (\{.*?\});\n", html, re.S)
    data = json.loads(m.group(1))
    assert len(data["contours"]) == len(results)
    assert all(len(c["t"]) <= 30 for c in data["contours"])
    assert len(data["bounds"]) == 4


def test_viewer_caps_terms():
    results = _fourier_results(1)
    html = ui.epicycle_viewer_html(results, [r["path"] for r in results], n_terms=500,
                                   colormap="sakura", max_terms=12)
    data = json.loads(re.search(r"var D = (\{.*?\});\n", html, re.S).group(1))
    assert data["k"] == 12


@pytest.mark.parametrize("style", ["paper", "night"])
def test_poster_styles_render(style):
    results = _fourier_results(2)
    fig = plot_poster([r["path"] for r in results], config={"render": {"colormap": "aurora"}},
                      style=style)
    assert fig_to_png_bytes(fig, dpi=20)[:4] == b"\x89PNG"
