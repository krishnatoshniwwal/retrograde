"""
symbolic.py — Convert numeric coefficients into SymPy expressions and LaTeX strings.

Supports two modes:
1. Spline segments → piecewise polynomial expressions
2. Fourier coefficients → sum of complex exponentials
"""

from __future__ import annotations

import sympy as sp
from sympy import latex, Rational, pi, exp, I, cos, sin, re, im
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.piecewise import SplineSegment
    from src.fourier import EpicycleCoeff


# ---------------------------------------------------------------------------
# Spline → SymPy
# ---------------------------------------------------------------------------

def spline_segment_to_sympy(
    segment: "SplineSegment",
    var: sp.Symbol | None = None,
) -> sp.Expr:
    """Convert one SplineSegment to a SymPy polynomial expression in *var*.

    The polynomial is expressed in terms of (var - t_start) so the
    coefficients directly match the stored values.

    Args:
        segment: A SplineSegment with t_start, t_end, x_coeffs, y_coeffs.
        var: SymPy symbol to use (defaults to 't').

    Returns:
        Tuple (x_expr, y_expr) of SymPy polynomials.
    """
    t = var or sp.Symbol("t")
    dt = t - sp.Rational(segment.t_start).limit_denominator(1000)

    def _poly(coeffs):
        """Build ascending-power polynomial from coefficient list."""
        expr = sp.Integer(0)
        for power, coeff in enumerate(coeffs):
            rounded = float(coeff)
            # Use Rational approximation for cleaner output
            c = sp.Float(rounded, 6)
            expr += c * dt**power
        return sp.expand(expr)

    x_expr = _poly(segment.x_coeffs)
    y_expr = _poly(segment.y_coeffs)
    return x_expr, y_expr


def spline_segments_to_piecewise(
    segments: list["SplineSegment"],
    max_segments: int = 20,
) -> tuple[sp.Expr, sp.Expr]:
    """Build a SymPy Piecewise expression for x(t) and y(t).

    To keep the LaTeX output manageable, at most *max_segments* segments
    are included (the first ones, which are the most important).

    Args:
        segments: List of SplineSegment from piecewise.extract_segments().
        max_segments: Cap the number of segments in the symbolic output.

    Returns:
        (x_piecewise, y_piecewise) SymPy expressions.
    """
    t = sp.Symbol("t")
    segs = segments[:max_segments]

    x_pieces = []
    y_pieces = []
    for seg in segs:
        cond = (t >= seg.t_start) & (t < seg.t_end)
        x_expr, y_expr = spline_segment_to_sympy(seg, var=t)
        x_pieces.append((x_expr, cond))
        y_pieces.append((y_expr, cond))

    # Add an otherwise clause using the last segment
    if segs:
        last = segs[-1]
        x_last, y_last = spline_segment_to_sympy(last, var=t)
        x_pieces.append((x_last, True))
        y_pieces.append((y_last, True))

    return sp.Piecewise(*x_pieces), sp.Piecewise(*y_pieces)


# ---------------------------------------------------------------------------
# Fourier → SymPy
# ---------------------------------------------------------------------------

def fourier_to_sympy(
    coeffs: list["EpicycleCoeff"],
    n_terms: int = 10,
) -> sp.Expr:
    """Build a SymPy expression for the Fourier series sum (complex form).

    Returns a complex-valued expression in symbol 't' (interpreted as
    time ∈ [0, 2π]) representing x(t) + i*y(t).

    Args:
        coeffs: Sorted list of EpicycleCoeff (from fourier.compute_dft).
        n_terms: Number of terms to include (keep small for readability).

    Returns:
        SymPy complex expression.
    """
    t = sp.Symbol("t")
    terms = coeffs[:n_terms]
    N = len(coeffs)

    expr = sp.Integer(0)
    for c in terms:
        freq = c.freq if c.freq <= N // 2 else c.freq - N
        amp = sp.Float(c.amplitude, 6)
        phase = sp.Float(c.phase, 6)
        # A * exp(i*(freq*t + phase)) = A*(cos(freq*t+phase) + i*sin(freq*t+phase))
        phasor = amp * exp(I * (freq * t + phase))
        expr += phasor

    return expr


def fourier_real_imag(
    coeffs: list["EpicycleCoeff"],
    n_terms: int = 10,
) -> tuple[sp.Expr, sp.Expr]:
    """Return separate real (x) and imaginary (y) SymPy expressions.

    Args:
        coeffs: DFT coefficients.
        n_terms: Number of epicycle terms.

    Returns:
        (x_expr, y_expr) in terms of SymPy symbol 't'.
    """
    t = sp.Symbol("t")
    terms = coeffs[:n_terms]
    N = len(coeffs)

    x_expr = sp.Integer(0)
    y_expr = sp.Integer(0)

    for c in terms:
        freq = c.freq if c.freq <= N // 2 else c.freq - N
        amp = sp.Float(c.amplitude, 6)
        phase = sp.Float(c.phase, 6)
        x_expr += amp * cos(freq * t + phase)
        y_expr += amp * sin(freq * t + phase)

    return x_expr, y_expr


# ---------------------------------------------------------------------------
# LaTeX string generation
# ---------------------------------------------------------------------------

def expr_to_latex(expr: sp.Expr) -> str:
    """Convert a SymPy expression to a LaTeX string.

    Args:
        expr: Any SymPy expression.

    Returns:
        LaTeX string (without surrounding $$).
    """
    return latex(expr)


def spline_to_latex_lines(
    segments: list["SplineSegment"],
    contour_index: int = 0,
    max_segments: int = 10,
) -> list[str]:
    """Produce a list of LaTeX equation strings for a spline contour.

    Each returned string is one piece of the piecewise function,
    suitable for inclusion in an align* or cases environment.

    Args:
        segments: SplineSegment list.
        contour_index: Index for labelling (e.g. contour 0, 1, 2...).
        max_segments: Maximum pieces to output.

    Returns:
        List of LaTeX strings.
    """
    lines = []
    t = sp.Symbol("t")
    for i, seg in enumerate(segments[:max_segments]):
        x_expr, y_expr = spline_segment_to_sympy(seg, var=t)
        t_lo = f"{seg.t_start:.4f}"
        t_hi = f"{seg.t_end:.4f}"
        lines.append(
            rf"x_{{{contour_index},{i}}}(t) = {latex(x_expr)}, "
            rf"\quad t \in [{t_lo},\, {t_hi}]"
        )
        lines.append(
            rf"y_{{{contour_index},{i}}}(t) = {latex(y_expr)}, "
            rf"\quad t \in [{t_lo},\, {t_hi}]"
        )
    return lines


def fourier_to_latex_lines(
    coeffs: list["EpicycleCoeff"],
    n_terms: int = 8,
    contour_index: int = 0,
) -> list[str]:
    """Produce LaTeX lines for a Fourier series contour.

    Args:
        coeffs: DFT coefficients.
        n_terms: Epicycle terms to include.
        contour_index: Label index.

    Returns:
        List of LaTeX strings.
    """
    x_expr, y_expr = fourier_real_imag(coeffs, n_terms=n_terms)
    return [
        rf"x_{{{contour_index}}}(t) = {latex(x_expr)}",
        rf"y_{{{contour_index}}}(t) = {latex(y_expr)}",
    ]
