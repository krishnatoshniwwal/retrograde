"""
tests/test_preprocessing.py — Unit tests for src/preprocessing.py
"""
import numpy as np
import pytest
import cv2
from pathlib import Path
import tempfile
import os

from src.preprocessing import (
    load_image,
    to_grayscale,
    resize_image,
    denoise,
    detect_edges,
    preprocess_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bgr_image():
    """Return a small synthetic BGR image."""
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    # Draw a white rectangle so there are edges
    img[20:80, 30:120] = 255
    return img


@pytest.fixture
def sample_gray_image(sample_bgr_image):
    return cv2.cvtColor(sample_bgr_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def tmp_image_file(sample_bgr_image, tmp_path):
    path = tmp_path / "test_img.png"
    cv2.imwrite(str(path), sample_bgr_image)
    return path


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

def test_load_image_returns_array(tmp_image_file):
    img = load_image(tmp_image_file)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


def test_load_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/img.png")


def test_load_image_invalid_file(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(bad)


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------

def test_grayscale_output_shape(sample_bgr_image):
    gray = to_grayscale(sample_bgr_image)
    assert gray.ndim == 2
    assert gray.shape == sample_bgr_image.shape[:2]


def test_grayscale_passthrough_if_already_gray(sample_gray_image):
    result = to_grayscale(sample_gray_image)
    assert result.shape == sample_gray_image.shape


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------

def test_resize_reduces_longest_side():
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    resized = resize_image(img, max_dim=100)
    assert max(resized.shape[:2]) == 100


def test_resize_preserves_aspect_ratio():
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    resized = resize_image(img, max_dim=200)
    h, w = resized.shape[:2]
    assert abs(h / w - 2.0) < 0.05  # 2:1 aspect preserved


def test_resize_no_upscale_if_small():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    resized = resize_image(img, max_dim=512)
    assert resized.shape == img.shape


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------

def test_denoise_output_shape(sample_gray_image):
    result = denoise(sample_gray_image, kernel_size=5)
    assert result.shape == sample_gray_image.shape
    assert result.dtype == sample_gray_image.dtype


def test_denoise_accepts_even_kernel(sample_gray_image):
    # Even kernel should be incremented to odd internally
    result = denoise(sample_gray_image, kernel_size=4)
    assert result.shape == sample_gray_image.shape


# ---------------------------------------------------------------------------
# detect_edges
# ---------------------------------------------------------------------------

def test_edges_binary(sample_gray_image):
    edges = detect_edges(sample_gray_image)
    unique = set(np.unique(edges).tolist())
    assert unique.issubset({0, 255})


def test_edges_detects_rectangle(sample_gray_image):
    edges = detect_edges(sample_gray_image, low=30, high=100)
    # There should be some non-zero pixels (the rectangle's edges)
    assert np.any(edges > 0)


# ---------------------------------------------------------------------------
# preprocess_array
# ---------------------------------------------------------------------------

def test_preprocess_array_returns_tuple(sample_bgr_image):
    edges, original = preprocess_array(sample_bgr_image)
    assert isinstance(edges, np.ndarray)
    assert isinstance(original, np.ndarray)
    assert edges.ndim == 2
    assert original.ndim == 3


def test_preprocess_array_with_config(sample_bgr_image):
    config = {"image": {"max_dim": 64, "denoise_kernel": 3},
              "edge": {"canny_low": 30, "canny_high": 90}}
    edges, original = preprocess_array(sample_bgr_image, config=config)
    assert max(original.shape[:2]) <= 64
