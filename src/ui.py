"""
ui.py — HTML building blocks for the Streamlit front-end.

Everything visual that isn't a Streamlit widget lives here: the animated
night-sky hero, pixel-art hills, HUD stat tiles, section headers, empty
states, colormap swatches and the in-browser epicycle viewer.

All functions return HTML strings; ``app.py`` decides where they go.
Markup is emitted as single-line-per-element with no blank lines, because
Streamlit's Markdown renderer ends a raw-HTML block at the first blank line
and treats indented lines as code.
"""

from __future__ import annotations

import html as _html
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.theme import NIGHT, colormap_hex_stops

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(markup: str) -> str:
    """Strip indentation and blank lines so Markdown keeps it as raw HTML."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(text: Any) -> str:
    """HTML-escape any value."""
    return _html.escape(str(text))


def load_css() -> str:
    """Return the global stylesheet wrapped in a <style> tag."""
    css = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    return f"<style>{css}</style>"


# ---------------------------------------------------------------------------
# Decorative pieces: stars, hills, epicycle machine
# ---------------------------------------------------------------------------

def _stars(n: int = 46, seed: int = 7) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.uniform(1, 99)
        y = rng.uniform(2, 70)
        size = rng.choice((2, 2, 2, 3, 4))
        delay = rng.uniform(0, 6)
        dur = rng.uniform(2.5, 5.5)
        color = rng.choice((NIGHT["fg"], NIGHT["fg"], NIGHT["moon"], NIGHT["accent"]))
        out.append(
            f'<i class="rg-star" style="left:{x:.1f}%;top:{y:.1f}%;width:{size}px;'
            f'height:{size}px;background:{color};animation-delay:{delay:.2f}s;'
            f'animation-duration:{dur:.2f}s"></i>'
        )
    return "".join(out)


def _stepped_ridge(width: int, base: float, amp: float, step: int, seed: int,
                   freq: float) -> str:
    """A pixel-stepped hill silhouette as an SVG path (crisp right angles)."""
    rng = random.Random(seed)
    phase = rng.uniform(0, math.tau)
    d = [f"M0 100 L0 {base:.0f}"]
    for x in range(0, width + step, step):
        y = base - amp * (0.55 * math.sin(x * freq + phase)
                          + 0.3 * math.sin(x * freq * 2.7 + phase * 1.7)
                          + 0.15 * math.sin(x * freq * 6.1))
        y = round(y / 4) * 4  # snap to a 4-unit pixel grid
        d.append(f"V{y:.0f} H{x + step}")
    d.append(f"V100 Z")
    return " ".join(d)


def hills_svg(height: int = 90, cls: str = "rg-hills") -> str:
    """Two layered pixel-art hill ridges, for the bottom of hero/footer."""
    far = _stepped_ridge(1200, 58, 22, 24, seed=3, freq=0.006)
    near = _stepped_ridge(1200, 78, 14, 16, seed=11, freq=0.011)
    return (
        f'<svg class="{cls}" viewBox="0 0 1200 100" preserveAspectRatio="none" '
        f'style="height:{height}px" shape-rendering="crispEdges" aria-hidden="true">'
        f'<path d="{far}" fill="{NIGHT["sky_1"]}" opacity="0.85"/>'
        f'<path d="{near}" fill="{NIGHT["bg_deep"]}"/>'
        f'</svg>'
    )


#: Hero epicycle machine: (frequency, radius, phase).  The first two terms are
#: the geocentric orbit of an outer planet — a genuine retrograde loop.
_HERO_TERMS: list[tuple[int, float, float]] = [
    (1, 1.9, 0.0),
    (6, 1.0, math.pi),
    (-4, 0.22, 0.6),
]


def epicycle_machine_svg(terms: list[tuple[int, float, float]] = _HERO_TERMS,
                         period: float = 26.0, size: int = 300,
                         cls: str = "rg-machine") -> str:
    """Self-animating SVG of nested rotating circles tracing their curve.

    Each circle is a nested ``<g>`` spun by CSS at its frequency *relative*
    to its parent, so the absolute angle of arm *k* is ``f_k·t + φ_k`` — the
    tip therefore rides exactly along the faint pre-computed trace.
    """
    reach = sum(r for _, r, _ in terms)
    s = 88.0 / reach

    # Pre-computed trace in SVG coordinates
    pts = []
    for i in range(721):
        t = math.tau * i / 720
        x = sum(r * math.cos(f * t + p) for f, r, p in terms) * s
        y = sum(r * math.sin(f * t + p) for f, r, p in terms) * s
        pts.append(f"{x:.2f},{y:.2f}")
    trace = " ".join(pts)

    # Nested rotating groups
    open_tags, close_tags = [], []
    prev_f, prev_p = 0, 0.0
    for i, (f, r, p) in enumerate(terms):
        rs = r * s
        a0 = math.degrees(p - prev_p)
        turn = 360.0 * (f - prev_f)
        open_tags.append(
            f'<g class="rg-spin" style="--a0:{a0:.2f}deg;--turn:{turn:.1f}deg;'
            f'animation-duration:{period}s">'
            f'<circle r="{rs:.2f}" class="rg-orbit rg-orbit-{i}"/>'
            f'<line x1="0" y1="0" x2="{rs:.2f}" y2="0" class="rg-arm"/>'
            f'<rect x="-1.5" y="-1.5" width="3" height="3" class="rg-joint"/>'
            f'<g transform="translate({rs:.2f} 0)">'
        )
        close_tags.append("</g></g>")
        prev_f, prev_p = f, p
    tip = '<rect x="-3" y="-3" width="6" height="6" class="rg-tip"/>'

    return (
        f'<svg class="{cls}" viewBox="-100 -100 200 200" width="{size}" height="{size}" '
        f'aria-label="Rotating circles tracing a retrograde loop">'
        f'<polyline points="{trace}" class="rg-trace"/>'
        f'{"".join(open_tags)}{tip}{"".join(reversed(close_tags))}'
        f'</svg>'
    )


def logo_mark(size: int = 30) -> str:
    """Tiny static epicycle glyph used next to the wordmark."""
    c = NIGHT
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" shape-rendering="crispEdges" aria-hidden="true">'
        f'<rect x="2" y="2" width="28" height="28" fill="{c["bg_deep"]}" stroke="{c["border"]}" stroke-width="2"/>'
        f'<circle cx="14" cy="16" r="8" fill="none" stroke="{c["accent"]}" stroke-width="2"/>'
        f'<circle cx="22" cy="16" r="4" fill="none" stroke="{c["sakura"]}" stroke-width="2"/>'
        f'<rect x="24" y="14" width="4" height="4" fill="{c["primary"]}"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def hero() -> str:
    """The animated night-sky banner at the top of the page."""
    return _flat(f"""
    <section class="rg-hero">
      <div class="rg-sky">{_stars()}<i class="rg-moon"></i>
        <i class="rg-petal" style="left:12%;animation-delay:0s"></i>
        <i class="rg-petal" style="left:38%;animation-delay:4s"></i>
        <i class="rg-petal" style="left:64%;animation-delay:8s"></i>
      </div>
      <div class="rg-hero-inner">
        <div class="rg-hero-copy">
          <div class="rg-kicker">&#9670; image &rarr; mathematics</div>
          <h1 class="rg-title">Retrograde</h1>
          <div class="rg-tagline">Every picture is a sum of circles.</div>
          <div class="rg-lede">Drop in a photo or sketch. Retrograde traces its outlines and
          solves for the exact equations &mdash; spinning Fourier epicycles or smooth cubic
          splines &mdash; that draw it back from nothing.</div>
          <div class="rg-chips">
            <span class="rg-chip rg-chip-lime">Fourier epicycles</span>
            <span class="rg-chip rg-chip-cyan">Cubic splines</span>
            <span class="rg-chip rg-chip-pink">LaTeX &middot; Desmos &middot; SVG</span>
          </div>
        </div>
        <div class="rg-hero-art">{epicycle_machine_svg()}</div>
      </div>
      {hills_svg(70)}
    </section>
    """)


def section_header(title: str, kicker: str = "", note: str = "") -> str:
    """A pixel-style section heading with a diamond marker and rule."""
    k = f'<span class="rg-sec-kicker">{esc(kicker)}</span>' if kicker else ""
    n = f'<p class="rg-sec-note">{note}</p>' if note else ""
    return _flat(f"""
    <div class="rg-sec">
      <div class="rg-sec-row">{k}<h3 class="rg-sec-title">{esc(title)}</h3><span class="rg-sec-rule"></span></div>
      {n}
    </div>
    """)


def sidebar_brand() -> str:
    return _flat(f"""
    <div class="rg-brand">
      {logo_mark(34)}
      <div><div class="rg-brand-name">Retrograde</div>
      <div class="rg-brand-sub">image &rarr; function</div></div>
    </div>
    """)


def sidebar_step(num: int, label: str) -> str:
    return f'<div class="rg-step"><span class="rg-step-num">{num:02d}</span>{esc(label)}</div>'


def stat_tiles(items: Iterable[tuple[str, str, str]]) -> str:
    """HUD stat row.  ``items`` = (value, label, accent) where accent is one of
    ``lime``, ``cyan``, ``pink``, ``moon``."""
    tiles = "".join(
        f'<div class="rg-stat rg-acc-{acc}"><div class="rg-stat-val">{esc(v)}</div>'
        f'<div class="rg-stat-lbl">{esc(lbl)}</div></div>'
        for v, lbl, acc in items
    )
    return f'<div class="rg-stats">{tiles}</div>'


def frame_label(text: str, tag: str = "") -> str:
    t = f'<span class="rg-frame-tag">{esc(tag)}</span>' if tag else ""
    return f'<div class="rg-frame-label"><span>{esc(text)}</span>{t}</div>'


def callout(text: str, kind: str = "info") -> str:
    """Inline message box.  kind ∈ info | ok | warn."""
    icon = {"info": "&#9670;", "ok": "&#10003;", "warn": "!"}[kind]
    return f'<div class="rg-callout rg-callout-{kind}"><b>{icon}</b><span>{text}</span></div>'


def empty_state(title: str, body: str, glyph: str = "upload") -> str:
    """Large empty-state panel shown before any input is provided."""
    steps = _flat("""
    <div class="rg-howto">
      <div class="rg-howto-step"><span>01</span><b>Trace</b><p>Canny edge detection finds the outlines, then OpenCV walks them into ordered point chains.</p></div>
      <div class="rg-howto-step"><span>02</span><b>Solve</b><p>A Fast Fourier Transform turns each chain into rotating circles &mdash; or cubic splines stitch it smooth.</p></div>
      <div class="rg-howto-step"><span>03</span><b>Export</b><p>Take the math with you: LaTeX, PDF, a live Desmos graph, a GIF or a print-ready poster.</p></div>
    </div>
    """)
    return _flat(f"""
    <div class="rg-empty">
      <div class="rg-empty-art">{epicycle_machine_svg(size=150, cls="rg-machine rg-machine-sm")}</div>
      <div class="rg-empty-title">{esc(title)}</div>
      <div class="rg-empty-body">{body}</div>
    </div>
    """) + steps


def colormap_swatch(name: str) -> str:
    stops = ", ".join(colormap_hex_stops(name, 10))
    return (f'<div class="rg-swatch" style="background:linear-gradient(90deg,{stops})">'
            f'<span>{esc(name)}</span></div>')


def rms_meter(rms: float, k: int, n: int) -> str:
    pct = 100.0 * k / max(n, 1)
    return _flat(f"""
    <div class="rg-meter">
      <div class="rg-meter-top"><span>terms <b>{k}</b> / {n}</span><span>RMS error <b>{rms:.2f} px</b></span></div>
      <div class="rg-meter-bar"><i style="width:{pct:.1f}%"></i></div>
    </div>
    """)


def export_card_head(glyph: str, title: str, desc: str, fmt: str) -> str:
    return _flat(f"""
    <div class="rg-card-head">
      <div class="rg-card-glyph">{glyph}</div>
      <div class="rg-card-fmt">{esc(fmt)}</div>
    </div>
    <div class="rg-card-title">{esc(title)}</div>
    <div class="rg-card-desc">{desc}</div>
    """)


def footer() -> str:
    return _flat(f"""
    <footer class="rg-footer">
      {hills_svg(56, "rg-hills rg-hills-footer")}
      <div class="rg-footer-bar">
        <span class="rg-footer-brand">{logo_mark(20)} Retrograde</span>
        <span>every picture is a sum of circles</span>
        <span class="rg-footer-tech">numpy.fft &middot; scipy &middot; sympy &middot; opencv</span>
      </div>
    </footer>
    """)


# ---------------------------------------------------------------------------
# In-browser epicycle viewer
# ---------------------------------------------------------------------------

def _signed_freq(freq: int, n: int) -> int:
    return freq if freq <= n // 2 else freq - n


def epicycle_viewer_html(
    results: list[dict[str, Any]],
    paths: list[np.ndarray],
    n_terms: int,
    colormap: str,
    height: int = 560,
    max_terms: int = 160,
) -> str:
    """A self-contained HTML/canvas page that animates every contour's
    epicycles live in the browser (no server-side rendering).

    Args:
        results:   Fourier pipeline results (each has ``coeffs``).
        paths:     Reconstructed paths, used only for the view bounds.
        n_terms:   Terms per contour selected in the UI.
        colormap:  Colormap name; contours are coloured along it.
        height:    Canvas height in CSS pixels.
        max_terms: Hard cap on circles per contour, to keep it 60 fps.
    """
    k = max(1, min(n_terms, max_terms))
    colors = colormap_hex_stops(colormap, max(len(results), 2))
    contours = []
    for i, res in enumerate(results):
        coeffs = res.get("coeffs", [])
        n = len(coeffs)
        terms = [
            [int(_signed_freq(int(c.freq), n)), round(float(c.amplitude), 4),
             round(float(c.phase), 5)]
            for c in coeffs[:k]
        ]
        if terms:
            contours.append({"t": terms, "c": colors[i % len(colors)]})

    if paths:
        pts = np.vstack([p for p in paths if len(p)])
        bounds = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                  float(pts[:, 0].max()), float(pts[:, 1].max())]
    else:
        bounds = [0.0, 0.0, 1.0, 1.0]

    data = json.dumps({"contours": contours, "bounds": bounds, "k": k})
    c = NIGHT
    template = (ASSETS_DIR / "epicycles.html").read_text(encoding="utf-8")
    return (template
            .replace("__DATA__", data)
            .replace("__HEIGHT__", str(height))
            .replace("__BG_DEEP__", c["bg_deep"])
            .replace("__BG__", c["bg"])
            .replace("__CARD__", c["card"])
            .replace("__BORDER__", c["border"])
            .replace("__FG__", c["fg"])
            .replace("__MUTED__", c["muted"])
            .replace("__PRIMARY__", c["primary"])
            .replace("__ACCENT__", c["accent"])
            .replace("__SAKURA__", c["sakura"])
            .replace("__SKY1__", c["sky_1"]))
