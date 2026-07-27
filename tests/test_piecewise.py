"""
tests/test_piecewise.py — Unit tests for src/piecewise.py
"""
import numpy as np
import pytest

from src.piecewise import (
    fit_spline,
    evaluate_spline,
    extract_segments,
    spline_pipeline,
    ParametricSpline,
    SplineSegment,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_sine_points(N: int = 50) -> np.ndarray:
    """N points along y = sin(x) for x in [0, 2π]."""
    t = np.linspace(0, 2 * np.pi, N)
    return np.column_stack([t, np.sin(t)])


def make_circle_points(N: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([50 * np.cos(t), 50 * np.sin(t)])


@pytest.fixture
def sine_pts():
    return make_sine_points(50)


@pytest.fixture
def circle_pts():
    return make_circle_points(64)


# ---------------------------------------------------------------------------
# fit_spline
# ---------------------------------------------------------------------------

def test_fit_spline_returns_parametric_spline(sine_pts):
    spline = fit_spline(sine_pts)
    assert isinstance(spline, ParametricSpline)


def test_fit_spline_requires_3_points():
    with pytest.raises(ValueError):
        fit_spline(np.array([[0, 0], [1, 1]]))


def test_fit_spline_passes_through_endpoints(sine_pts):
    spline = fit_spline(sine_pts)
    t0 = spline.t_knots[0]
    t1 = spline.t_knots[-1]
    x0 = float(spline.cs_x(t0))
    y0 = float(spline.cs_y(t0))
    x1 = float(spline.cs_x(t1))
    y1 = float(spline.cs_y(t1))
    np.testing.assert_allclose(x0, sine_pts[0, 0], atol=1e-5)
    np.testing.assert_allclose(y0, sine_pts[0, 1], atol=1e-5)
    np.testing.assert_allclose(x1, sine_pts[-1, 0], atol=1e-5)
    np.testing.assert_allclose(y1, sine_pts[-1, 1], atol=1e-5)


def test_fit_spline_passes_through_all_knots(sine_pts):
    """The spline should interpolate all input points exactly."""
    spline = fit_spline(sine_pts)
    t = spline.t_knots
    x_eval = spline.cs_x(t)
    y_eval = spline.cs_y(t)
    np.testing.assert_allclose(x_eval, sine_pts[:, 0], atol=1e-5)
    np.testing.assert_allclose(y_eval, sine_pts[:, 1], atol=1e-5)


def test_fit_spline_deduplicates_points():
    """Duplicate points should not cause a crash."""
    pts = np.array([[0, 0], [0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    spline = fit_spline(pts)
    assert isinstance(spline, ParametricSpline)


# ---------------------------------------------------------------------------
# evaluate_spline
# ---------------------------------------------------------------------------

def test_evaluate_spline_shape(sine_pts):
    spline = fit_spline(sine_pts)
    path = evaluate_spline(spline, n_points=200)
    assert path.shape == (200, 2)


def test_evaluate_spline_smooth(sine_pts):
    """Consecutive points should not jump wildly."""
    spline = fit_spline(sine_pts)
    path = evaluate_spline(spline, n_points=500)
    diffs = np.diff(path, axis=0)
    max_step = np.max(np.hypot(diffs[:, 0], diffs[:, 1]))
    # For a sine over [0, 2π], steps should be small
    assert max_step < 0.5


# ---------------------------------------------------------------------------
# extract_segments
# ---------------------------------------------------------------------------

def test_extract_segments_count(sine_pts):
    spline = fit_spline(sine_pts)
    segs = extract_segments(spline)
    # Should have len(knots) - 1 segments
    assert len(segs) == len(spline.t_knots) - 1


def test_extract_segments_structure(sine_pts):
    spline = fit_spline(sine_pts)
    segs = extract_segments(spline)
    for seg in segs:
        assert isinstance(seg, SplineSegment)
        assert len(seg.x_coeffs) == 4
        assert len(seg.y_coeffs) == 4
        assert seg.t_start < seg.t_end


def test_extract_segments_covers_full_range(sine_pts):
    spline = fit_spline(sine_pts)
    segs = extract_segments(spline)
    assert abs(segs[0].t_start - spline.t_knots[0]) < 1e-10
    assert abs(segs[-1].t_end - spline.t_knots[-1]) < 1e-10


def test_extract_segments_cached(sine_pts):
    """Second call should return the same object."""
    spline = fit_spline(sine_pts)
    segs1 = extract_segments(spline)
    segs2 = extract_segments(spline)
    assert segs1 is segs2


# ---------------------------------------------------------------------------
# spline_pipeline
# ---------------------------------------------------------------------------

def test_spline_pipeline_keys(circle_pts):
    result = spline_pipeline(circle_pts)
    assert "spline" in result
    assert "path" in result
    assert "segments" in result


def test_spline_pipeline_path_shape(circle_pts):
    result = spline_pipeline(circle_pts)
    assert result["path"].shape[1] == 2


def test_spline_pipeline_with_config(circle_pts):
    config = {"spline": {"n_knots": 20}}
    result = spline_pipeline(circle_pts, config=config)
    assert len(result["segments"]) > 0


def test_spline_pipeline_reconstruction_error(circle_pts):
    """Reconstructed path should be reasonably close to the original circle."""
    result = spline_pipeline(circle_pts, config={"spline": {"n_knots": 40}})
    path = result["path"]
    # All reconstructed points should be within 10% of radius=50
    dists = np.hypot(path[:, 0], path[:, 1])
    np.testing.assert_allclose(dists, 50.0, atol=5.0)  # within 10%
