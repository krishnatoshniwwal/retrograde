"""
tests/test_fourier.py — Unit tests for src/fourier.py

Key mathematical properties verified:
  1. DFT of a pure cosine/sine has energy at exactly one frequency.
  2. Reconstruction of a circle (single rotating vector) is accurate.
  3. Increasing n_terms strictly reduces reconstruction error.
  4. EpicycleCoeff amplitudes are non-negative and sorted descending.
"""
import numpy as np
import pytest

from src.fourier import (
    compute_dft,
    reconstruct_path,
    get_epicycle_frames,
    dft_pipeline,
    EpicycleCoeff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle(N: int = 128, radius: float = 50.0) -> np.ndarray:
    """Generate N points evenly sampled from a circle of given radius."""
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.column_stack([x, y])


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared distance between two paths (resampled to same length)."""
    from scipy.interpolate import interp1d
    n = min(len(original), len(reconstructed))
    t_orig = np.linspace(0, 1, len(original))
    t_new = np.linspace(0, 1, n)
    ox = interp1d(t_orig, original[:, 0])(t_new)
    oy = interp1d(t_orig, original[:, 1])(t_new)
    rx = interp1d(np.linspace(0, 1, len(reconstructed)), reconstructed[:, 0])(t_new)
    ry = interp1d(np.linspace(0, 1, len(reconstructed)), reconstructed[:, 1])(t_new)
    return float(np.mean((ox - rx)**2 + (oy - ry)**2))


# ---------------------------------------------------------------------------
# compute_dft
# ---------------------------------------------------------------------------

def test_dft_returns_correct_count():
    signal = np.ones(64, dtype=complex)
    coeffs = compute_dft(signal)
    assert len(coeffs) == 64


def test_dft_amplitudes_nonnegative():
    pts = make_circle(64)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    for c in coeffs:
        assert c.amplitude >= 0.0


def test_dft_sorted_descending():
    pts = make_circle(64)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    amps = [c.amplitude for c in coeffs]
    assert amps == sorted(amps, reverse=True)


def test_dft_circle_dominant_frequency():
    """A circle sampled at N points should have one dominant frequency."""
    N = 64
    pts = make_circle(N, radius=100.0)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    # The top coefficient should carry most of the energy
    top_amp = coeffs[0].amplitude
    total_amp = sum(c.amplitude for c in coeffs)
    # Top term should carry > 90% of total amplitude for a pure circle
    assert top_amp / total_amp > 0.9


def test_dft_dc_component_for_offset_signal():
    """A constant signal should have all energy at frequency 0."""
    N = 32
    # Constant complex signal → DC only
    signal = np.full(N, 5.0 + 3.0j, dtype=complex)
    coeffs = compute_dft(signal)
    dc = next(c for c in coeffs if c.freq == 0)
    assert abs(dc.real - 5.0) < 1e-6
    assert abs(dc.imag - 3.0) < 1e-6


def test_dft_coeff_fields():
    signal = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    coeffs = compute_dft(signal)
    for c in coeffs:
        assert isinstance(c, EpicycleCoeff)
        assert hasattr(c, "freq")
        assert hasattr(c, "amplitude")
        assert hasattr(c, "phase")


# ---------------------------------------------------------------------------
# reconstruct_path
# ---------------------------------------------------------------------------

def test_reconstruct_path_shape():
    pts = make_circle(64)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    path = reconstruct_path(coeffs, n_terms=10, n_points=200)
    assert path.shape == (200, 2)


def test_reconstruct_circle_accuracy():
    """Reconstruct a circle — error should be < 5% of radius²."""
    N = 128
    radius = 50.0
    pts = make_circle(N, radius=radius)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    path = reconstruct_path(coeffs, n_terms=N, n_points=N)
    err = reconstruction_error(pts, path)
    assert err < 0.05 * radius**2


def test_more_terms_reduces_error():
    """Adding more Fourier terms should reduce (or maintain) error."""
    pts = make_circle(64)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)

    errors = []
    for n_terms in [5, 10, 30, 60]:
        path = reconstruct_path(coeffs, n_terms=n_terms, n_points=200)
        errors.append(reconstruction_error(pts, path))

    # Errors should be non-increasing
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] - 1e-3  # allow tiny floating point wiggle


# ---------------------------------------------------------------------------
# get_epicycle_frames
# ---------------------------------------------------------------------------

def test_get_epicycle_frames_count():
    pts = make_circle(32)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    frames = get_epicycle_frames(coeffs, n_terms=5, n_frames=20)
    assert len(frames) == 20


def test_get_epicycle_frames_structure():
    pts = make_circle(32)
    signal = pts[:, 0] + 1j * pts[:, 1]
    coeffs = compute_dft(signal)
    frames = get_epicycle_frames(coeffs, n_terms=5, n_frames=10)
    for frame in frames:
        assert len(frame) == 5
        for item in frame:
            assert len(item) == 3  # (cx, cy, radius)


# ---------------------------------------------------------------------------
# dft_pipeline
# ---------------------------------------------------------------------------

def test_dft_pipeline_keys():
    pts = make_circle(64)
    result = dft_pipeline(pts)
    assert "coeffs" in result
    assert "path" in result
    assert "n_terms" in result


def test_dft_pipeline_with_config():
    pts = make_circle(64)
    config = {"fourier": {"n_terms": 20}}
    result = dft_pipeline(pts, config=config)
    assert result["n_terms"] == 20
    assert result["path"].shape[1] == 2
