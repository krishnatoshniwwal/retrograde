"""
fourier.py — DFT-based Fourier / epicycle representation.

Converts a contour (sequence of complex points) into Fourier coefficients
via numpy.fft, then supports reconstruction and epicycle animation data.

Math summary
------------
Each 2-D contour point (x_k, y_k) is encoded as z_k = x_k + i*y_k.
The DFT decomposes this signal into N complex frequencies:

    Z_n = Σ_{k=0}^{N-1}  z_k * exp(-2πi*n*k / N)

Each term corresponds to a circle (epicycle) rotating at frequency n,
with amplitude |Z_n|/N and phase angle arg(Z_n).

Keeping only the M coefficients with the largest amplitudes gives an
M-epicycle approximation of the original path.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpicycleCoeff:
    """One Fourier / epicycle term."""
    freq: int          # integer frequency index
    amplitude: float   # |Z_n| / N
    phase: float       # arg(Z_n) in radians
    real: float        # Re(Z_n) / N  (useful for reconstruction)
    imag: float        # Im(Z_n) / N


# ---------------------------------------------------------------------------
# Core DFT computation
# ---------------------------------------------------------------------------

def compute_dft(signal: np.ndarray) -> list[EpicycleCoeff]:
    """Compute the DFT of a complex-valued signal and return sorted coefficients.

    The coefficients are sorted in descending order of amplitude so that
    taking the first M gives the best M-term approximation.

    Args:
        signal: (N,) complex128 array representing the contour.

    Returns:
        List of EpicycleCoeff objects sorted by amplitude descending.
    """
    N = len(signal)
    X = np.fft.fft(signal)  # vectorised FFT — O(N log N)

    coeffs: list[EpicycleCoeff] = []
    for n in range(N):
        z = X[n] / N
        coeffs.append(
            EpicycleCoeff(
                freq=n,
                amplitude=abs(z),
                phase=float(np.angle(z)),
                real=float(z.real),
                imag=float(z.imag),
            )
        )

    # Sort by amplitude descending so top-M captures the most energy
    coeffs.sort(key=lambda c: c.amplitude, reverse=True)
    return coeffs


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(
    coeffs: list[EpicycleCoeff],
    n_terms: int,
    n_points: int = 1000,
) -> np.ndarray:
    """Reconstruct a 2-D path from the top-M Fourier coefficients.

    Args:
        coeffs: List of EpicycleCoeff (e.g. from compute_dft).
        n_terms: Number of terms (epicycles) to include.
        n_points: Number of sample points along the reconstructed path.

    Returns:
        (n_points, 2) float64 array of (x, y) coordinates.
    """
    terms = coeffs[:n_terms]
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Sum of rotating phasors: each coeff contributes
    # A * exp(i * (freq * t + phase))  for continuous t in [0, 2π)
    signal = np.zeros(n_points, dtype=np.complex128)
    N_signal = len(coeffs)  # original signal length determines frequency scaling

    for c in terms:
        # Map freq index back to signed frequency for correct reconstruction
        freq = c.freq if c.freq <= N_signal // 2 else c.freq - N_signal
        phasor = c.amplitude * np.exp(1j * (freq * t + c.phase))
        signal += phasor

    x = signal.real
    y = signal.imag
    return np.column_stack([x, y])


def compute_rms_error(
    path_approx: np.ndarray,
    path_full: np.ndarray,
) -> float:
    """Compute the RMS distance between two reconstructed paths.

    Both paths are resampled to the same number of points before comparison,
    so they don't need to have equal length.

    Args:
        path_approx: (M, 2) approximate path (fewer terms).
        path_full:   (N, 2) reference path (full terms).

    Returns:
        RMS pointwise distance in pixels (float).
    """
    n = min(len(path_approx), len(path_full))
    # Resample both to n points via linear indexing
    idx_approx = np.round(np.linspace(0, len(path_approx) - 1, n)).astype(int)
    idx_full   = np.round(np.linspace(0, len(path_full)   - 1, n)).astype(int)
    a = path_approx[idx_approx]
    b = path_full[idx_full]
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def get_epicycle_frames(
    coeffs: list[EpicycleCoeff],
    n_terms: int,
    n_frames: int = 300,
) -> list[list[tuple[float, float, float]]]:
    """Compute per-frame epicycle arm positions for animation.

    For each time step t, returns the chain of (cx, cy, radius) values
    describing each rotating circle's centre and radius.

    Args:
        coeffs: Sorted DFT coefficients.
        n_terms: Number of epicycles to show.
        n_frames: Total number of animation frames.

    Returns:
        List of frames; each frame is a list of (cx, cy, radius) tuples,
        one per epicycle, in drawing order (outermost to innermost).
    """
    terms = coeffs[:n_terms]
    N_signal = len(coeffs)
    t_values = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    frames = []
    for t in t_values:
        x, y = 0.0, 0.0
        frame_circles: list[tuple[float, float, float]] = []
        for c in terms:
            freq = c.freq if c.freq <= N_signal // 2 else c.freq - N_signal
            angle = freq * t + c.phase
            x += c.amplitude * np.cos(angle)
            y += c.amplitude * np.sin(angle)
            frame_circles.append((x, y, c.amplitude))
        frames.append(frame_circles)

    return frames


# ---------------------------------------------------------------------------
# Convenience pipeline function
# ---------------------------------------------------------------------------

def dft_pipeline(
    contour_pts: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full Fourier pipeline on a (N, 2) contour.

    Args:
        contour_pts: (N, 2) float array of (x, y) contour points.
        config: Optional config dict (uses fourier.n_terms).

    Returns:
        Dict with keys:
          - 'coeffs': list[EpicycleCoeff]
          - 'path': (M, 2) reconstructed path
          - 'n_terms': int number of terms used
    """
    cfg = (config or {}).get("fourier", {})
    n_terms: int = cfg.get("n_terms", 100)

    signal = contour_pts[:, 0] + 1j * contour_pts[:, 1]
    coeffs = compute_dft(signal)

    n_terms = min(n_terms, len(coeffs))
    path = reconstruct_path(coeffs, n_terms=n_terms)

    return {
        "coeffs": coeffs,
        "path": path,
        "n_terms": n_terms,
    }
