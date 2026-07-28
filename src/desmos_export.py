"""
desmos_export.py — Convert pipeline results into Desmos-compatible output.

Two export formats:
  1. JSON expression list  — a list of {"id", "latex", "color"} dicts that
     Desmos can import directly (paste into the expression panel).
  2. Self-contained HTML  — embeds the real Desmos GraphingCalculator JS API
     and programmatically injects every expression so the file opens as a
     fully interactive, zoomable graph.

Why we do NOT use SymPy's latex() here
---------------------------------------
Desmos has its own LaTeX parser.  SymPy produces notation it rejects
(e.g. \\operatorname{re}, \\left/\\right auto-sizing, nested \\frac inside
trig arguments).  We build the LaTeX strings with direct string templates so
they stay inside the subset Desmos actually understands.

Parametric curve format in Desmos
----------------------------------
  (x_{i}(t), y_{i}(t))
with a domain restriction appended:  \\left\\{0 \\le t \\le 2\\pi\\right\\}

The helper functions x_{i}(t) and y_{i}(t) are defined as separate
expressions so the sidebar stays readable.
"""

from __future__ import annotations

import json
import math
from typing import Any

from src.fourier import EpicycleCoeff
from src.piecewise import SplineSegment


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# A curated neon palette that looks good on Desmos's white background.
# Cycles if there are more contours than colors.
_NEON_PALETTE = [
    "#ff003c",  # neon red
    "#00f0ff",  # cyan
    "#ff8c00",  # orange
    "#a259ff",  # violet
    "#00ff9f",  # mint
    "#ffde00",  # yellow
    "#ff69b4",  # hot pink
    "#4fc3f7",  # sky blue
    "#76ff03",  # lime
    "#ff4081",  # pink
    "#18ffff",  # aqua
    "#ffd740",  # amber
]


def _contour_color(index: int) -> str:
    """Return a hex color string for contour at *index*, cycling the palette."""
    return _NEON_PALETTE[index % len(_NEON_PALETTE)]


# ---------------------------------------------------------------------------
# Fourier → Desmos expressions
# ---------------------------------------------------------------------------

def fourier_to_desmos_exprs(
    coeffs: list[EpicycleCoeff],
    n_terms: int,
    contour_idx: int,
) -> list[dict[str, str]]:
    """Convert Fourier coefficients into Desmos expression dicts.

    Produces three entries per contour:
      - x definition:  x_{i}(t) = Σ A·cos(freq·t + phase)
      - y definition:  y_{i}(t) = Σ A·sin(freq·t + phase)
      - parametric curve: (x_{i}(t), y_{i}(t)) with domain 0 ≤ t ≤ 2π

    Args:
        coeffs: Sorted list of EpicycleCoeff (from fourier.compute_dft).
        n_terms: Number of epicycle terms to include.
        contour_idx: Index used for Desmos variable subscripts.

    Returns:
        List of {"id", "latex", "color"} dicts.
    """
    terms = coeffs[:n_terms]
    N_signal = len(coeffs)
    color = _contour_color(contour_idx)
    i = contour_idx

    x_parts: list[str] = []
    y_parts: list[str] = []

    for c in terms:
        freq = c.freq if c.freq <= N_signal // 2 else c.freq - N_signal
        amp = c.amplitude
        phase = c.phase

        if abs(amp) < 1e-9:
            continue

        amp_s = f"{amp:.6g}"
        phase_s = f"{phase:.6g}"

        if freq == 0:
            x_parts.append(f"{amp_s}\\cos\\left({phase_s}\\right)")
            y_parts.append(f"{amp_s}\\sin\\left({phase_s}\\right)")
        else:
            freq_s = str(freq) if freq != 1 else ""
            freq_neg_s = str(freq) if freq != -1 else "-"

            if freq > 0:
                arg = f"{freq}t+{phase_s}" if phase_s != "0" else f"{freq}t"
            else:
                arg = f"{freq}t+{phase_s}" if phase_s != "0" else f"{freq}t"

            x_parts.append(f"{amp_s}\\cos\\left({arg}\\right)")
            y_parts.append(f"{amp_s}\\sin\\left({arg}\\right)")

    x_latex_rhs = "+".join(x_parts) if x_parts else "0"
    y_latex_rhs = "+".join(y_parts) if y_parts else "0"

    # Desmos function definitions use subscript notation: x_{0}(t)
    # NOTE: image y-axis points downward; Desmos y-axis points upward.
    # Negate y so the rendered shape is the right way up.
    x_def = f"x_{{f{i}}}\\left(t\\right)={x_latex_rhs}"
    y_def = f"y_{{f{i}}}\\left(t\\right)=-\\left({y_latex_rhs}\\right)"

    # Parametric curve with domain restriction
    curve = (
        f"\\left(x_{{f{i}}}\\left(t\\right),\\ y_{{f{i}}}\\left(t\\right)\\right)"
        f"\\left\\{{0\\le t\\le2\\pi\\right\\}}"
    )

    return [
        {"id": f"xf{i}", "latex": x_def, "color": color, "hidden": True},
        {"id": f"yf{i}", "latex": y_def, "color": color, "hidden": True},
        {"id": f"cf{i}", "latex": curve, "color": color},
    ]


# ---------------------------------------------------------------------------
# Spline → Desmos expressions
# ---------------------------------------------------------------------------

def spline_to_desmos_exprs(
    segments: list[SplineSegment],
    contour_idx: int,
    max_segs: int = 30,
) -> list[dict[str, str]]:
    """Convert spline segments into Desmos expression dicts.

    Each segment produces one parametric entry with a domain restriction
    so that only the relevant piece of the curve is drawn.

    Args:
        segments: List of SplineSegment from piecewise.extract_segments().
        contour_idx: Index used for Desmos variable subscripts.
        max_segs: Maximum segments to export (Desmos can slow with too many).

    Returns:
        List of {"id", "latex", "color"} dicts.
    """
    color = _contour_color(contour_idx)
    i = contour_idx
    exprs: list[dict[str, str]] = []

    for seg_idx, seg in enumerate(segments[:max_segs]):
        t0 = seg.t_start
        t1 = seg.t_end

        def _poly_latex(coeffs: Any, var: str = "t", t_start: float = 0.0) -> str:
            """Build a polynomial latex string from ascending coefficients."""
            parts = []
            for power, coeff in enumerate(coeffs):
                c = float(coeff)
                if abs(c) < 1e-12:
                    continue
                c_s = f"{c:.6g}"
                if power == 0:
                    parts.append(c_s)
                elif power == 1:
                    if t_start == 0.0:
                        parts.append(f"{c_s}{var}")
                    else:
                        t0_s = f"{t_start:.6g}"
                        parts.append(f"{c_s}\\left({var}-{t0_s}\\right)")
                else:
                    if t_start == 0.0:
                        parts.append(f"{c_s}{var}^{{{power}}}")
                    else:
                        t0_s = f"{t_start:.6g}"
                        parts.append(
                            f"{c_s}\\left({var}-{t0_s}\\right)^{{{power}}}"
                        )
            return "+".join(parts) if parts else "0"

        x_rhs = _poly_latex(seg.x_coeffs, var="t", t_start=t0)
        y_rhs = _poly_latex(seg.y_coeffs, var="t", t_start=t0)

        # Negate y so image coords (y downward) map correctly to Desmos (y upward)
        y_rhs_neg = f"-\\left({y_rhs}\\right)" if y_rhs != "0" else "0"

        t0_s = f"{t0:.6g}"
        t1_s = f"{t1:.6g}"
        domain = f"\\left\\{{{t0_s}\\le t\\le{t1_s}\\right\\}}"
        curve = f"\\left({x_rhs},\\ {y_rhs_neg}\\right){domain}"

        exprs.append({
            "id": f"cs{i}_{seg_idx}",
            "latex": curve,
            "color": color,
        })

    return exprs


# ---------------------------------------------------------------------------
# Build the full expression list
# ---------------------------------------------------------------------------

def build_desmos_expression_list(
    results: list[dict[str, Any]],
    method: str = "fourier",
    n_terms: int = 80,
    max_contours: int = 50,
    max_segs_per_contour: int = 25,
) -> list[dict[str, str]]:
    """Build the complete Desmos expression list from all pipeline results.

    Args:
        results: List of pipeline result dicts (from dft_pipeline or
                 spline_pipeline), one per contour.
        method: "fourier" or "piecewise".
        n_terms: Fourier terms per contour (ignored for piecewise).
        max_contours: Maximum number of contours to include.
        max_segs_per_contour: Maximum spline segments per contour (piecewise only).

    Returns:
        List of Desmos expression dicts {"id", "latex", "color"}.
    """
    all_exprs: list[dict[str, str]] = []

    for idx, res in enumerate(results[:max_contours]):
        if method == "fourier":
            coeffs = res.get("coeffs", [])
            if not coeffs:
                continue
            exprs = fourier_to_desmos_exprs(coeffs, n_terms=n_terms, contour_idx=idx)
        else:
            segs = res.get("segments", [])
            if not segs:
                continue
            exprs = spline_to_desmos_exprs(
                segs, contour_idx=idx, max_segs=max_segs_per_contour
            )

        all_exprs.extend(exprs)

    return all_exprs


# ---------------------------------------------------------------------------
# Serialise to JSON
# ---------------------------------------------------------------------------

def expression_list_to_desmos_state(
    exprs: list[dict[str, str]],
    viewport: dict | None = None,
) -> str:
    """Serialise expressions as a Desmos native save-state JSON.

    This is the format desmos.com expects when you use:
      Hamburger menu → Load Graph → choose this file.

    Each expression is wrapped with ``"type": "expression"`` and the whole
    object is wrapped in the standard Desmos state envelope
    (``version``, ``graph``, ``expressions``).

    Args:
        exprs: List of expression dicts from build_desmos_expression_list.
        viewport: Optional dict with keys xmin, xmax, ymin, ymax.  A
                  reasonable default is supplied if omitted.

    Returns:
        Pretty-printed JSON string, directly loadable at desmos.com/calculator.
    """
    vp = viewport or {"xmin": -400, "ymin": -400, "xmax": 400, "ymax": 400}

    state = {
        "version": 9,
        "graph": {
            "viewport": vp,
            "showGrid": False,
            "showXAxis": False,
            "showYAxis": False,
        },
        "expressions": {
            "list": [
                {
                    "type": "expression",
                    "id": e["id"],
                    "color": e["color"],
                    "latex": e["latex"],
                    **(  # propagate optional fields cleanly
                        {"hidden": True} if e.get("hidden") else {}
                    ),
                }
                for e in exprs
            ]
        },
    }
    return json.dumps(state, indent=2, ensure_ascii=False)


def expression_list_to_json(exprs: list[dict[str, str]]) -> str:
    """Serialise the raw Desmos API expression list (for programmatic use).

    This format is suitable for ``calculator.setExpression()`` calls via the
    Desmos JS API.  It is NOT directly importable via the desmos.com web UI.
    Use :func:`expression_list_to_desmos_state` for that.

    Args:
        exprs: List of expression dicts.

    Returns:
        JSON string.
    """
    return json.dumps(exprs, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Build interactive HTML
# ---------------------------------------------------------------------------

def build_desmos_html(
    exprs: list[dict[str, str]],
    title: str = "Image-to-Function — Desmos",
) -> str:
    """Generate a self-contained HTML file embedding the Desmos API.

    The file uses the official Desmos GraphingCalculator JS API
    (api.desmos.com/v1.9/calculator.js) to inject every expression
    programmatically. Opening the file in any modern browser gives a
    fully interactive, zoomable graph with all equations visible in the
    sidebar and a legend showing per-contour colors.

    Args:
        exprs: List of Desmos expression dicts (from build_desmos_expression_list).
        title: HTML page title and graph label.

    Returns:
        HTML string (UTF-8 safe).
    """
    exprs_json = json.dumps(exprs, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta name="description" content="Interactive Desmos graph generated by retrograde image-to-function renderer."/>
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #ffffff;
    color: #1a1a1a;
    font-family: 'Courier New', monospace;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.2rem;
    border-bottom: 1px solid #d0d0d0;
    background: #f5f5f5;
    flex-shrink: 0;
  }}
  header .logo {{
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1a1a1a;
  }}
  header .meta {{
    font-size: 0.7rem;
    color: #888;
    letter-spacing: 0.08em;
  }}
  #calc-container {{
    flex: 1;
    position: relative;
  }}
  #calculator {{
    width: 100%;
    height: 100%;
  }}
  #loading {{
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #ffffff;
    gap: 1rem;
    z-index: 10;
    transition: opacity 0.4s ease;
  }}
  #loading .spinner {{
    width: 40px; height: 40px;
    border: 3px solid #e0e0e0;
    border-top-color: #333;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }}
  #loading p {{ font-size: 0.8rem; color: #888; letter-spacing: 0.1em; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<header>
  <div class="logo">[ RETROGRADE ] // DESMOS_EXPORT</div>
  <div class="meta">EXPR_COUNT: {len(exprs)} &nbsp;|&nbsp; INTERACTIVE_MODE</div>
</header>
<div id="calc-container">
  <div id="loading">
    <div class="spinner"></div>
    <p>LOADING_GRAPH...</p>
  </div>
  <div id="calculator"></div>
</div>
<script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
<script>
(function () {{
  var exprs = {exprs_json};

  var elt = document.getElementById('calculator');
  var calc = Desmos.GraphingCalculator(elt, {{
    expressionWidth: 240,
    settingsMenu: true,
    zoomButtons: true,
    expressions: true,
    keypad: false,
    border: false,
    // Start with a viewport large enough to contain pixel-space image coords
    // (images are resized to max ~512px; contour coords sit in that range)
    mathBounds: {{ left: -600, right: 1100, bottom: -900, top: 600 }},
  }});

  // White background for Desmos canvas
  calc.updateSettings({{ backgroundColor: '#ffffff' }});

  // Inject all expressions
  exprs.forEach(function(expr) {{
    var entry = {{ id: expr.id, latex: expr.latex, color: expr.color }};
    if (expr.hidden) {{ entry.hidden = true; }}
    calc.setExpression(entry);
  }});

  // zoomFit after expressions have had time to evaluate.
  // Parametric curves need a moment; we try at 1 s and again at 3 s.
  function tryZoomFit() {{
    try {{ calc.zoomFit(); }} catch(e) {{}}
  }}
  setTimeout(tryZoomFit, 1000);
  setTimeout(tryZoomFit, 3000);

  // Hide loading overlay after the first zoomFit attempt
  var loading = document.getElementById('loading');
  setTimeout(function() {{
    loading.style.opacity = '0';
    setTimeout(function() {{ loading.style.display = 'none'; }}, 400);
  }}, 1200);
}})();
</script>
</body>
</html>"""

    return html
