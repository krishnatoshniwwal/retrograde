"""
preprocessing.py — Stage 1 of the Image-to-Function pipeline.

Handles: image loading, grayscale conversion, resizing, denoising,
and Canny edge detection.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR numpy array.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as uint8 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR or already-grayscale image to a single-channel gray image.

    Args:
        img: Input image (BGR uint8 or gray uint8).

    Returns:
        Grayscale image (H, W) uint8.
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Resize image so its longest side equals *max_dim*, preserving aspect ratio.

    Args:
        img: Grayscale or colour image.
        max_dim: Target size for the longest dimension.

    Returns:
        Resized image.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before edge detection.

    Args:
        img: Grayscale image.
        kernel_size: Size of the Gaussian kernel (must be odd and positive).

    Returns:
        Blurred grayscale image.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def detect_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Run Canny edge detection on a grayscale image.

    Args:
        img: Grayscale (possibly blurred) image.
        low: Lower hysteresis threshold.
        high: Upper hysteresis threshold.

    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    return cv2.Canny(img, low, high)


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def preprocess(
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline on an image file.

    Returns a (edge_map, original_bgr) tuple where:
    - edge_map is a binary uint8 array (H, W) suitable for contour extraction.
    - original_bgr is the resized colour image for reference/display.

    Args:
        path: Path to the input image.
        config: Optional dict with keys from config/default.yaml.
                Uses sensible defaults if not supplied.

    Returns:
        (edge_map, original_bgr)
    """
    cfg_image = (config or {}).get("image", {})
    cfg_edge = (config or {}).get("edge", {})

    max_dim: int = cfg_image.get("max_dim", 512)
    denoise_kernel: int = cfg_image.get("denoise_kernel", 5)
    canny_low: int = cfg_edge.get("canny_low", 50)
    canny_high: int = cfg_edge.get("canny_high", 150)

    img = load_image(path)
    img = resize_image(img, max_dim=max_dim)
    original_bgr = img.copy()

    gray = to_grayscale(img)
    blurred = denoise(gray, kernel_size=denoise_kernel)
    edges = detect_edges(blurred, low=canny_low, high=canny_high)

    return edges, original_bgr


def preprocess_array(
    img_array: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as :func:`preprocess` but accepts a numpy array directly.

    Useful for the Streamlit UI which receives an already-loaded image.

    Args:
        img_array: BGR or RGB numpy array.
        config: Optional config dict.

    Returns:
        (edge_map, original_bgr)
    """
    cfg_image = (config or {}).get("image", {})
    cfg_edge = (config or {}).get("edge", {})

    max_dim: int = cfg_image.get("max_dim", 512)
    denoise_kernel: int = cfg_image.get("denoise_kernel", 5)
    canny_low: int = cfg_edge.get("canny_low", 50)
    canny_high: int = cfg_edge.get("canny_high", 150)

    img = resize_image(img_array, max_dim=max_dim)
    original_bgr = img.copy()

    # Handle RGB input from PIL/Streamlit
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = to_grayscale(img)

    blurred = denoise(gray, kernel_size=denoise_kernel)
    edges = detect_edges(blurred, low=canny_low, high=canny_high)

    return edges, original_bgr
