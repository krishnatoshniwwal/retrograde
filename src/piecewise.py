"""
piecewise.py — Piecewise cubic spline fitting for contours.

Fits a parametric cubic spline to an ordered sequence of (x, y) points
using scipy.interpolate, then evaluates the spline and extracts polynomial
segment coefficients for symbolic/LaTeX export.

Math summary
------------
Given N points {(x_k, y_k)} we parameterise by arc-length t ∈ [0, 1]
and fit two independent cubic splines:
    x(t) = spline_x(t)
    y(t) = spline_y(t)

Each spline piece over the interval [t_k, t_{k+1}] is a degree-3 polynomial:
    f(t) = a + b*(t-t_k) + c*(t-t_k)^2 + d*(t-t_k)^3
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplineSegment:
    """One polynomial piece of a parametric spline."""
    t_start: float       # start of the interval
    t_end: float         # end of the interval
    x_coeffs: np.ndarray  # [a, b, c, d] for x(t), ascending powers
    y_coeffs: np.ndarray  # [a, b, c, d] for y(t), ascending powers


@dataclass
class ParametricSpline:
    """A full parametric spline (x(t), y(t)) over [0, 1]."""
    cs_x: CubicSpline
    cs_y: CubicSpline
    t_knots: np.ndarray    # parameter values at original points
    segments: list[SplineSegment] = None  # filled lazily


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _arc_length_param(points: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-length parameterisation in [0, 1].

    Args:
        points: (N, 2) array of (x, y).

    Returns:
        (N,) array of parameter values in [0, 1].
    """
    diffs = np.diff(points, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total < 1e-10:
        return np.linspace(0, 1, len(points))
    return cumulative / total


def fit_spline(points: np.ndarray) -> ParametricSpline:
    """Fit a parametric cubic spline to a sequence of (x, y) points.

    Uses arc-length parameterisation to avoid artifacts when points
    are unevenly spaced.

    Args:
        points: (N, 2) float array, N >= 3.

    Returns:
        ParametricSpline object.
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points for spline, got {len(points)}")

    # Remove duplicate consecutive points to avoid singular spline
    deltas = np.diff(points, axis=0)
    keep = np.concatenate([[True], np.any(deltas != 0, axis=1)])
    points = points[keep]

    if len(points) < 3:
        raise ValueError("Insufficient unique points after deduplication.")

    t = _arc_length_param(points)
    cs_x = CubicSpline(t, points[:, 0], bc_type="not-a-knot")
    cs_y = CubicSpline(t, points[:, 1], bc_type="not-a-knot")

    return ParametricSpline(cs_x=cs_x, cs_y=cs_y, t_knots=t)


def evaluate_spline(spline: ParametricSpline, n_points: int = 1000) -> np.ndarray:
    """Evaluate the spline at evenly-spaced parameter values.

    Args:
        spline: ParametricSpline to evaluate.
        n_points: Number of sample points.

    Returns:
        (n_points, 2) array of (x, y).
    """
    t = np.linspace(spline.t_knots[0], spline.t_knots[-1], n_points)
    x = spline.cs_x(t)
    y = spline.cs_y(t)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Segment coefficient extraction
# ---------------------------------------------------------------------------

def extract_segments(spline: ParametricSpline) -> list[SplineSegment]:
    """Extract polynomial coefficients for each spline piece.

    scipy CubicSpline stores coefficients as descending powers.
    We convert to ascending powers [a, b, c, d] (constant first)
    to match standard piecewise notation.

    Args:
        spline: Fitted ParametricSpline.

    Returns:
        List of SplineSegment, one per interval between knots.
    """
    if spline.segments is not None:
        return spline.segments

    t = spline.t_knots
    # scipy's .c attribute: shape (4, N-1) in descending power order
    cx = spline.cs_x.c  # shape (4, n_intervals)
    cy = spline.cs_y.c

    segments = []
    for i in range(cx.shape[1]):
        # descending [d, c, b, a] → ascending [a, b, c, d]
        x_asc = cx[:, i][::-1]
        y_asc = cy[:, i][::-1]
        segments.append(
            SplineSegment(
                t_start=float(t[i]),
                t_end=float(t[i + 1]),
                x_coeffs=x_asc,
                y_coeffs=y_asc,
            )
        )

    spline.segments = segments
    return segments


# ---------------------------------------------------------------------------
# Convenience pipeline function
# ---------------------------------------------------------------------------

def spline_pipeline(
    contour_pts: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full spline pipeline on a (N, 2) contour.

    Args:
        contour_pts: (N, 2) float array of (x, y) contour points.
        config: Optional config dict (uses spline.n_knots for subsampling).

    Returns:
        Dict with keys:
          - 'spline': ParametricSpline
          - 'path': (M, 2) reconstructed path
          - 'segments': list[SplineSegment]
    """
    cfg = (config or {}).get("spline", {})
    n_knots: int = cfg.get("n_knots", 50)

    # Optionally subsample for very long contours to keep n_knots manageable
    if len(contour_pts) > n_knots * 3:
        idx = np.linspace(0, len(contour_pts) - 1, n_knots, dtype=int)
        pts = contour_pts[idx]
    else:
        pts = contour_pts

    spline = fit_spline(pts)
    path = evaluate_spline(spline, n_points=1000)
    segments = extract_segments(spline)

    return {
        "spline": spline,
        "path": path,
        "segments": segments,
    }
