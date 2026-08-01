"""
explainer.py — Static-tier "explain the math" panel (Tier B-3).

Generates Markdown explanations for each UI tab using template strings
filled with actual numbers from the current pipeline run.

All functions are pure (no side effects, no I/O) so they can be called
inside @st.cache_data wrappers or directly from the UI layer.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Reconstruction tab
# ---------------------------------------------------------------------------

def explain_reconstruction(
    n_contours: int,
    method: str,
    n_terms: int,
    colormap: str,
    canny_low: int = 50,
    canny_high: int = 150,
) -> str:
    """Return a Markdown explanation for the Reconstruction tab.

    Args:
        n_contours:  Number of contours extracted from the image.
        method:      \"Fourier (Epicycles)\" or \"Piecewise (Splines)\".
        n_terms:     Number of terms / knots used.
        colormap:    Matplotlib colormap name in use.
        canny_low:   Canny lower threshold.
        canny_high:  Canny upper threshold.
    """
    is_fourier = "Fourier" in method
    method_name = "Fourier series" if is_fourier else "cubic splines"
    term_noun   = "epicycles (rotating circles)" if is_fourier else "spline knots"

    contour_word = "contour" if n_contours == 1 else "contours"

    explanation = f"""
### How this reconstruction was produced

**Step 1 — Edge detection**

The image was passed through a **Canny edge detector** with thresholds
`low={canny_low}` and `high={canny_high}`. Canny computes image gradients
(how fast pixel brightness changes) and keeps only the pixels where the
gradient is strong and consistent — the sharp outlines you see as a binary
edge map. Higher thresholds catch only the boldest edges; lower thresholds
include fainter details.

**Step 2 — Contour extraction**

OpenCV's `findContours` traced the edge map and found **{n_contours} {contour_word}** —
ordered sequences of (x, y) points each describing one closed or open outline.
Short contours (noise artefacts) were discarded. The remaining {n_contours}
were simplified with the **Douglas–Peucker algorithm**, which removes redundant
points that lie nearly on a straight line, reducing the point count while
preserving the visual shape.

**Step 3 — Function fitting ({method_name})**

Each contour's point sequence was fed into the **{method}** pipeline
using **{n_terms} {term_noun}** per contour. The reconstruction above is the
plot of those mathematical functions — no pixel data, just equations evaluated
on a fine grid of parameter values.

**Step 4 — Rendering**

The reconstructed paths are drawn using matplotlib's `LineCollection` with
the **{colormap}** colormap, so colour shifts continuously along each curve
(position along the path maps to the colour gradient) rather than being a
single flat colour. The result is both visually readable and mathematically
exact.
""".strip()

    return explanation


# ---------------------------------------------------------------------------
# Epicycles tab
# ---------------------------------------------------------------------------

def explain_epicycles(
    coeffs_list: list[list[Any]],
    n_terms: int,
) -> str:
    """Return a Markdown explanation for the Epicycles tab.

    Args:
        coeffs_list: List of coefficient lists from dft_pipeline results
                     (each element is a list[EpicycleCoeff]).
        n_terms:     Number of epicycles shown in the animation.
    """
    if not coeffs_list:
        return "_No Fourier data available — run the pipeline in Fourier mode first._"

    # Use the first contour's coefficients as the example
    coeffs = coeffs_list[0]
    top_n  = coeffs[:n_terms]

    if not top_n:
        return "_No coefficients available._"

    largest  = top_n[0]
    smallest = top_n[-1]

    largest_freq  = abs(largest.freq)
    smallest_freq = abs(smallest.freq)
    largest_amp   = largest.amplitude
    smallest_amp  = smallest.amplitude
    largest_phase = math.degrees(largest.phase)

    total_contours = len(coeffs_list)
    contour_word   = "contour" if total_contours == 1 else "contours"

    explanation = f"""
### What you are watching

**The Fourier series as rotating circles**

Every closed curve in the plane can be written as a sum of rotating circles —
this is the geometric picture of a **Fourier series**. Each circle (epicycle)
spins at a fixed angular frequency; its radius is the amplitude of the
corresponding Fourier coefficient.

**The Discrete Fourier Transform (DFT)**

The pipeline encoded each contour as a sequence of **complex numbers**
`z_k = x_k + i·y_k` (the contour points), then applied the DFT:

```
Z_n = Σ z_k · exp(−2πi·n·k / N)
```

This decomposes the signal into N frequency components. The components are
sorted by amplitude (largest first) so that keeping the top {n_terms} captures
the most visual information.

**Numbers from this specific run** *(first contour, {n_terms} epicycles)*

| | Frequency | Amplitude | Phase |
|---|---|---|---|
| **Largest circle** | {largest_freq} | {largest_amp:.2f} px | {largest_phase:.1f}° |
| **Smallest circle** | {smallest_freq} | {smallest_amp:.4f} px | — |

The **largest circle** (amplitude ≈ **{largest_amp:.1f} px**) contributes the
coarse overall shape — it determines where the curve sits and its rough
proportions. The **smallest circle** (amplitude ≈ **{smallest_amp:.4f} px**)
adds fine detail that is almost invisible at normal zoom. Adding more epicycles
is mathematically equivalent to raising the image resolution of the
reconstruction.

**{total_contours} {contour_word} processed** — each has its own independent set of {n_terms}
epicycles playing simultaneously.
""".strip()

    return explanation


# ---------------------------------------------------------------------------
# Equations tab
# ---------------------------------------------------------------------------

def explain_equations(
    results: list[dict[str, Any]],
    method: str,
    contour_idx: int = 0,
    term_idx: int = 0,
) -> str:
    """Return a Markdown explanation for the Equations tab.

    Args:
        results:      Pipeline results list (each dict has 'coeffs' or 'segments').
        method:       \"Fourier (Epicycles)\" or \"Piecewise (Splines)\".
        contour_idx:  Which contour to explain (0-based).
        term_idx:     Which term/segment to highlight (0-based).
    """
    is_fourier = "Fourier" in method

    if not results or contour_idx >= len(results):
        return "_No equation data available._"

    res = results[contour_idx]

    if is_fourier:
        coeffs = res.get("coeffs", [])
        if not coeffs:
            return "_No Fourier coefficients found for this contour._"

        c = coeffs[min(term_idx, len(coeffs) - 1)]
        amp   = c.amplitude
        phase = math.degrees(c.phase)
        freq  = c.freq
        n_total = len(coeffs)

        explanation = f"""
### Reading a Fourier term (Contour {contour_idx + 1}, term {term_idx + 1})

The equation panel shows the **parametric Fourier series**:

```
x(t) + i·y(t)  =  Σ  Aₙ · exp(i·(fₙ·t + φₙ))
```

Each term is one rotating circle. Here is what the highlighted term means:

| Property | Value | Meaning |
|---|---|---|
| **Frequency fₙ** | {freq} | This circle completes {abs(freq)} full rotation{"s" if abs(freq) != 1 else ""} per drawing cycle |
| **Amplitude Aₙ** | {amp:.4f} px | The radius of this circle is ~{amp:.2f} pixels |
| **Phase φₙ** | {phase:.2f}° | The circle starts rotated {phase:.1f}° from the positive x-axis at t=0 |
| **Rank** | {term_idx + 1} of {n_total} | Sorted by amplitude — this is the #{term_idx + 1} most important term |

**Why amplitude matters:** Amplitude controls how much this term shifts the
curve. Large-amplitude terms (the first few) set the overall shape; small
amplitudes (later terms) refine fine edges. Truncating at {n_total} terms
means all smaller details are intentionally dropped — this is **lossy
compression** in function space.

**Why frequency matters:** Frequency determines how rapidly this circle
rotates. Low frequencies (small |fₙ|) produce smooth, broad curves. High
frequencies produce tight, rapid wiggles — they encode sharp corners and
fine texture.
""".strip()

    else:
        segments = res.get("segments", [])
        if not segments:
            return "_No spline segments found for this contour._"

        seg = segments[min(term_idx, len(segments) - 1)]
        t0 = seg.t_start
        t1 = seg.t_end
        cx = list(seg.x_coeffs) if seg.x_coeffs is not None else []
        cy = list(seg.y_coeffs) if seg.y_coeffs is not None else []

        explanation = f"""
### Reading a spline segment (Contour {contour_idx + 1}, segment {term_idx + 1})

The equations panel shows a **piecewise cubic spline** — a chain of
polynomial pieces, each valid over a small parameter interval.

This segment covers parameter interval **t ∈ [{t0:.4f}, {t1:.4f}]**.

On this interval the curve is:

```
x(t) = a₀ + a₁·(t−t₀) + a₂·(t−t₀)² + a₃·(t−t₀)³
y(t) = b₀ + b₁·(t−t₀) + b₂·(t−t₀)² + b₃·(t−t₀)³
```

{"**x-coefficients:** " + ", ".join(f"{v:.4f}" for v in cx[:4]) if cx else ""}
{"**y-coefficients:** " + ", ".join(f"{v:.4f}" for v in cy[:4]) if cy else ""}

**Why piecewise?** A single global polynomial through many points produces
violent oscillations (Runge's phenomenon). Splines avoid this by using many
short low-degree pieces, each fitted locally, stitched together so the
curve is smooth at the joins.

**There are {len(segments)} segments** in this contour — together they form
the giant piecewise function that traces the full outline.
""".strip()

    return explanation
