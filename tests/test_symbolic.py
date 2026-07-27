"""
tests/test_symbolic.py — Unit tests for src/symbolic.py
"""
import numpy as np
import pytest
import sympy as sp

from src.symbolic import (
    spline_segment_to_sympy,
    spline_segments_to_piecewise,
    fourier_to_sympy,
    fourier_real_imag,
    expr_to_latex,
    spline_to_latex_lines,
    fourier_to_latex_lines,
)
from src.piecewise import fit_spline, extract_segments, SplineSegment
from src.fourier import compute_dft, EpicycleCoeff


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_sine_spline_segments(N: int = 20):
    t = np.linspace(0, 2 * np.pi, N)
    pts = np.column_stack([t, np.sin(t)])
    spline = fit_spline(pts)
    return extract_segments(spline)


def make_fourier_coeffs(N: int = 32):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([50 * np.cos(t), 50 * np.sin(t)])
    signal = pts[:, 0] + 1j * pts[:, 1]
    return compute_dft(signal)


@pytest.fixture
def sine_segments():
    return make_sine_spline_segments()


@pytest.fixture
def circle_coeffs():
    return make_fourier_coeffs()


# ---------------------------------------------------------------------------
# spline_segment_to_sympy
# ---------------------------------------------------------------------------

def test_segment_to_sympy_returns_tuple(sine_segments):
    seg = sine_segments[0]
    result = spline_segment_to_sympy(seg)
    assert isinstance(result, tuple)
    assert len(result) == 2
    x_expr, y_expr = result
    assert isinstance(x_expr, sp.Expr)
    assert isinstance(y_expr, sp.Expr)


def test_segment_to_sympy_custom_variable(sine_segments):
    seg = sine_segments[0]
    s = sp.Symbol("s")
    x_expr, y_expr = spline_segment_to_sympy(seg, var=s)
    assert s in x_expr.free_symbols or x_expr.is_number


def test_segment_to_sympy_is_polynomial(sine_segments):
    seg = sine_segments[0]
    t = sp.Symbol("t")
    x_expr, _ = spline_segment_to_sympy(seg, var=t)
    # Should be a polynomial in t (or a constant)
    poly = sp.Poly(x_expr, t) if not x_expr.is_number else None
    if poly is not None:
        assert poly.degree() <= 3


# ---------------------------------------------------------------------------
# spline_segments_to_piecewise
# ---------------------------------------------------------------------------

def test_piecewise_x_is_sympy_piecewise(sine_segments):
    x_pw, y_pw = spline_segments_to_piecewise(sine_segments)
    assert isinstance(x_pw, sp.Piecewise)
    assert isinstance(y_pw, sp.Piecewise)


def test_piecewise_respects_max_segments(sine_segments):
    x_pw, _ = spline_segments_to_piecewise(sine_segments, max_segments=3)
    # Piecewise should have at most max_segments+1 args (last is "otherwise")
    assert len(x_pw.args) <= 4  # 3 + 1 otherwise


# ---------------------------------------------------------------------------
# fourier_to_sympy
# ---------------------------------------------------------------------------

def test_fourier_to_sympy_is_expr(circle_coeffs):
    expr = fourier_to_sympy(circle_coeffs, n_terms=5)
    assert isinstance(expr, sp.Expr)


def test_fourier_to_sympy_contains_t(circle_coeffs):
    t = sp.Symbol("t")
    expr = fourier_to_sympy(circle_coeffs, n_terms=5)
    # Should contain the symbol t (unless all terms are zero, which won't happen)
    assert t in expr.free_symbols or expr.is_number


# ---------------------------------------------------------------------------
# fourier_real_imag
# ---------------------------------------------------------------------------

def test_fourier_real_imag_returns_two_exprs(circle_coeffs):
    x_expr, y_expr = fourier_real_imag(circle_coeffs, n_terms=5)
    assert isinstance(x_expr, sp.Expr)
    assert isinstance(y_expr, sp.Expr)


def test_fourier_real_imag_n_terms_1(circle_coeffs):
    """Single-term series should produce a valid expression."""
    x_expr, y_expr = fourier_real_imag(circle_coeffs, n_terms=1)
    assert isinstance(x_expr, sp.Expr)


# ---------------------------------------------------------------------------
# expr_to_latex
# ---------------------------------------------------------------------------

def test_expr_to_latex_nonempty(circle_coeffs):
    x_expr, _ = fourier_real_imag(circle_coeffs, n_terms=3)
    latex_str = expr_to_latex(x_expr)
    assert isinstance(latex_str, str)
    assert len(latex_str) > 0


def test_expr_to_latex_constant():
    expr = sp.Integer(42)
    assert expr_to_latex(expr) == "42"


# ---------------------------------------------------------------------------
# spline_to_latex_lines
# ---------------------------------------------------------------------------

def test_spline_to_latex_lines_returns_list(sine_segments):
    lines = spline_to_latex_lines(sine_segments, contour_index=0, max_segments=3)
    assert isinstance(lines, list)
    assert len(lines) > 0


def test_spline_to_latex_lines_are_strings(sine_segments):
    lines = spline_to_latex_lines(sine_segments, contour_index=0)
    for line in lines:
        assert isinstance(line, str)
        assert len(line) > 0


def test_spline_to_latex_lines_max_segments(sine_segments):
    lines = spline_to_latex_lines(sine_segments, contour_index=0, max_segments=2)
    # 2 segments × 2 lines (x, y) = 4 lines max
    assert len(lines) <= 4


# ---------------------------------------------------------------------------
# fourier_to_latex_lines
# ---------------------------------------------------------------------------

def test_fourier_to_latex_lines_returns_two_lines(circle_coeffs):
    lines = fourier_to_latex_lines(circle_coeffs, n_terms=5, contour_index=0)
    assert len(lines) == 2  # one for x(t), one for y(t)


def test_fourier_to_latex_lines_contain_label(circle_coeffs):
    lines = fourier_to_latex_lines(circle_coeffs, n_terms=5, contour_index=7)
    assert "7" in lines[0]  # contour index appears in the label
