"""
contours.py — Stage 2 & 3 of the Image-to-Function pipeline.

Handles: contour extraction from edge maps, simplification via
Douglas-Peucker, and conversion to complex-number sequences for DFT.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours(
    edge_map: np.ndarray,
    min_length: int = 20,
) -> list[np.ndarray]:
    """Extract ordered point sequences from a binary edge map.

    Uses RETR_LIST (all contours) with CHAIN_APPROX_NONE (every pixel)
    to get dense, accurate point sequences.

    Args:
        edge_map: Binary uint8 image (H, W), e.g. output of Canny.
        min_length: Discard contours with fewer points than this.

    Returns:
        List of (N, 2) float32 arrays, each representing one contour
        as a sequence of (x, y) coordinates.
    """
    raw, _ = cv2.findContours(
        edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    
    h, w = edge_map.shape
    cx, cy = w / 2.0, h / 2.0
    
    result = []
    for contour in raw:
        pts = contour.reshape(-1, 2).astype(np.float32)
        # Shift coordinates to be centered around the origin (0,0)
        pts -= np.array([cx, cy], dtype=np.float32)
        if len(pts) >= min_length:
            result.append(pts)
    # Sort by length descending so the most prominent contours come first
    result.sort(key=len, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def simplify_contour(
    contour: np.ndarray,
    epsilon_fraction: float = 0.002,
) -> np.ndarray:
    """Reduce point count via the Douglas-Peucker algorithm.

    Args:
        contour: (N, 2) array of (x, y) float32 points.
        epsilon_fraction: Tolerance as a fraction of the contour arc length.
                          Smaller → more points kept, larger → more reduction.

    Returns:
        Simplified (M, 2) array where M <= N.
    """
    arc_length = cv2.arcLength(contour.astype(np.float32), closed=False)
    epsilon = epsilon_fraction * arc_length
    simplified = cv2.approxPolyDP(
        contour.astype(np.float32), epsilon, closed=False
    )
    return simplified.reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Conversion to complex representation
# ---------------------------------------------------------------------------

def contour_to_complex(contour: np.ndarray) -> np.ndarray:
    """Convert a (N, 2) contour to a complex-valued 1-D array.

    Each point (x, y) maps to the complex number x + iy.
    This is the input format expected by the DFT / Fourier module.

    Args:
        contour: (N, 2) array of (x, y) float64/float32.

    Returns:
        (N,) complex128 array.
    """
    pts = contour.astype(np.float64)
    return pts[:, 0] + 1j * pts[:, 1]


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def get_all_contours(
    edge_map: np.ndarray,
    config: dict[str, Any] | None = None,
    simplify: bool = True,
) -> list[np.ndarray]:
    """Extract, filter, and optionally simplify all contours from an edge map.

    Args:
        edge_map: Binary uint8 edge image.
        config: Optional config dict (uses keys contour.min_length,
                contour.epsilon_fraction).
        simplify: Whether to apply Douglas-Peucker simplification.

    Returns:
        List of (N, 2) float32 arrays.
    """
    cfg = (config or {}).get("contour", {})
    min_length: int = cfg.get("min_length", 20)
    epsilon_fraction: float = cfg.get("epsilon_fraction", 0.002)

    contours = extract_contours(edge_map, min_length=min_length)

    if simplify:
        contours = [
            simplify_contour(c, epsilon_fraction=epsilon_fraction)
            for c in contours
        ]
        # Re-filter: simplification may have removed points below min_length
        contours = [c for c in contours if len(c) >= max(3, min_length // 4)]

    return contours
