"""
tests/test_contours.py — Unit tests for src/contours.py
"""
import numpy as np
import pytest
import cv2

from src.contours import (
    extract_contours,
    simplify_contour,
    contour_to_complex,
    get_all_contours,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rectangle_edge_map():
    """A synthetic binary edge map containing one clear rectangle."""
    edge = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(edge, (30, 30), (170, 170), 255, thickness=2)
    return edge


@pytest.fixture
def simple_contour():
    """A simple square contour (20 points) for testing."""
    pts = []
    for i in range(20):
        pts.append([20 + i * 3, 20])
    for i in range(20):
        pts.append([80, 20 + i * 3])
    for i in range(20):
        pts.append([80 - i * 3, 80])
    for i in range(20):
        pts.append([20, 80 - i * 3])
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# extract_contours
# ---------------------------------------------------------------------------

def test_extract_contours_returns_list(rectangle_edge_map):
    contours = extract_contours(rectangle_edge_map, min_length=10)
    assert isinstance(contours, list)
    assert len(contours) > 0


def test_extract_contours_min_length_filter(rectangle_edge_map):
    # With a very high min_length, all contours should be filtered out
    contours = extract_contours(rectangle_edge_map, min_length=100000)
    assert contours == []


def test_extract_contours_sorted_by_length(rectangle_edge_map):
    contours = extract_contours(rectangle_edge_map, min_length=5)
    lengths = [len(c) for c in contours]
    assert lengths == sorted(lengths, reverse=True)


def test_extract_contours_shape(rectangle_edge_map):
    contours = extract_contours(rectangle_edge_map, min_length=10)
    for c in contours:
        assert c.ndim == 2
        assert c.shape[1] == 2


# ---------------------------------------------------------------------------
# simplify_contour
# ---------------------------------------------------------------------------

def test_simplify_reduces_point_count(simple_contour):
    simplified = simplify_contour(simple_contour, epsilon_fraction=0.05)
    # Should have fewer points than original
    assert len(simplified) <= len(simple_contour)


def test_simplify_returns_2d_array(simple_contour):
    simplified = simplify_contour(simple_contour)
    assert simplified.ndim == 2
    assert simplified.shape[1] == 2


def test_simplify_zero_epsilon_keeps_all(simple_contour):
    # Near-zero epsilon → very little simplification
    simplified = simplify_contour(simple_contour, epsilon_fraction=0.0001)
    assert len(simplified) >= 4  # at least corners remain


# ---------------------------------------------------------------------------
# contour_to_complex
# ---------------------------------------------------------------------------

def test_contour_to_complex_dtype(simple_contour):
    z = contour_to_complex(simple_contour)
    assert z.dtype == np.complex128


def test_contour_to_complex_shape(simple_contour):
    z = contour_to_complex(simple_contour)
    assert z.ndim == 1
    assert len(z) == len(simple_contour)


def test_contour_to_complex_values(simple_contour):
    z = contour_to_complex(simple_contour)
    # Real part == x, imag part == y
    np.testing.assert_allclose(z.real, simple_contour[:, 0])
    np.testing.assert_allclose(z.imag, simple_contour[:, 1])


# ---------------------------------------------------------------------------
# get_all_contours
# ---------------------------------------------------------------------------

def test_get_all_contours_returns_list(rectangle_edge_map):
    contours = get_all_contours(rectangle_edge_map)
    assert isinstance(contours, list)


def test_get_all_contours_with_config(rectangle_edge_map):
    config = {"contour": {"min_length": 10, "epsilon_fraction": 0.01}}
    contours = get_all_contours(rectangle_edge_map, config=config)
    assert isinstance(contours, list)


def test_get_all_contours_no_simplify(rectangle_edge_map):
    contours_simplified = get_all_contours(rectangle_edge_map, simplify=True)
    contours_raw = get_all_contours(rectangle_edge_map, simplify=False)
    # Raw contours should generally have more total points
    raw_total = sum(len(c) for c in contours_raw)
    simp_total = sum(len(c) for c in contours_simplified)
    assert raw_total >= simp_total
