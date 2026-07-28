"""
tests/test_desmos_export.py — Unit tests for src/desmos_export.py.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from src.fourier import EpicycleCoeff
from src.piecewise import SplineSegment
from src.desmos_export import (
    fourier_to_desmos_exprs,
    spline_to_desmos_exprs,
    build_desmos_expression_list,
    expression_list_to_desmos_state,
    expression_list_to_json,
    build_desmos_html,
    _contour_color,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_coeffs(n: int = 5) -> list[EpicycleCoeff]:
    """Create synthetic Fourier coefficients for testing."""
    coeffs = []
    for k in range(n):
        coeffs.append(
            EpicycleCoeff(
                freq=k,
                amplitude=1.0 / (k + 1),
                phase=0.1 * k,
                real=math.cos(0.1 * k) / (k + 1),
                imag=math.sin(0.1 * k) / (k + 1),
            )
        )
    return coeffs


def _make_segments(n: int = 3) -> list[SplineSegment]:
    """Create synthetic spline segments for testing."""
    segs = []
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        segs.append(
            SplineSegment(
                t_start=t0,
                t_end=t1,
                x_coeffs=np.array([float(i), 1.0, 0.5, 0.1]),
                y_coeffs=np.array([float(i) * 0.5, 0.8, 0.2, 0.05]),
            )
        )
    return segs


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

def test_contour_color_cycles():
    """_contour_color should cycle through the palette without IndexError."""
    for i in range(50):
        color = _contour_color(i)
        assert color.startswith("#"), f"Expected hex color, got {color!r}"
        assert len(color) == 7, f"Expected 7-char hex color, got {color!r}"


def test_contour_color_distinct():
    """Adjacent indices should produce different colors."""
    colors = [_contour_color(i) for i in range(12)]
    assert len(set(colors)) == 12, "First 12 colors should all be distinct"


# ---------------------------------------------------------------------------
# Fourier → Desmos
# ---------------------------------------------------------------------------

class TestFourierToDesmos:
    def test_returns_three_entries(self):
        """Should produce exactly 3 expression dicts per contour."""
        coeffs = _make_coeffs(5)
        exprs = fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=0)
        assert len(exprs) == 3

    def test_entry_keys(self):
        """Each expression dict must have 'id', 'latex', 'color'."""
        coeffs = _make_coeffs(5)
        for expr in fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=0):
            assert "id" in expr
            assert "latex" in expr
            assert "color" in expr

    def test_curve_entry_has_no_hidden(self):
        """The parametric curve entry (third) should not be hidden."""
        coeffs = _make_coeffs(5)
        exprs = fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=0)
        curve = exprs[2]
        assert not curve.get("hidden", False)

    def test_helper_defs_are_hidden(self):
        """x/y helper definitions (first two) should be hidden."""
        coeffs = _make_coeffs(5)
        exprs = fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=0)
        assert exprs[0].get("hidden") is True
        assert exprs[1].get("hidden") is True

    def test_curve_contains_domain_restriction(self):
        """Parametric curve latex should include a domain restriction."""
        coeffs = _make_coeffs(5)
        exprs = fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=0)
        curve_latex = exprs[2]["latex"]
        assert "2\\pi" in curve_latex, "Expected domain '0 ≤ t ≤ 2π' in latex"

    def test_unique_ids_across_contours(self):
        """Expression IDs should be unique when called for different contours."""
        coeffs = _make_coeffs(5)
        ids_0 = {e["id"] for e in fourier_to_desmos_exprs(coeffs, 3, contour_idx=0)}
        ids_1 = {e["id"] for e in fourier_to_desmos_exprs(coeffs, 3, contour_idx=1)}
        assert ids_0.isdisjoint(ids_1), "IDs must not collide across contours"

    def test_contour_color_applied(self):
        """All three expressions for the same contour share the same color."""
        coeffs = _make_coeffs(5)
        exprs = fourier_to_desmos_exprs(coeffs, n_terms=3, contour_idx=2)
        colors = {e["color"] for e in exprs}
        assert len(colors) == 1, "All expressions for one contour must share one color"


# ---------------------------------------------------------------------------
# Spline → Desmos
# ---------------------------------------------------------------------------

class TestSplineToDesmos:
    def test_one_entry_per_segment(self):
        """Should produce one expression dict per segment (up to max_segs)."""
        segs = _make_segments(4)
        exprs = spline_to_desmos_exprs(segs, contour_idx=0, max_segs=4)
        assert len(exprs) == 4

    def test_max_segs_respected(self):
        """max_segs should cap the number of expressions."""
        segs = _make_segments(10)
        exprs = spline_to_desmos_exprs(segs, contour_idx=0, max_segs=3)
        assert len(exprs) == 3

    def test_domain_restriction_present(self):
        """Each segment latex must include a domain restriction."""
        segs = _make_segments(2)
        for expr in spline_to_desmos_exprs(segs, contour_idx=0):
            assert "\\le t\\le" in expr["latex"], \
                f"No domain restriction in: {expr['latex']!r}"

    def test_unique_ids(self):
        """Each segment in a contour should have a unique ID."""
        segs = _make_segments(5)
        exprs = spline_to_desmos_exprs(segs, contour_idx=0)
        ids = [e["id"] for e in exprs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# build_desmos_expression_list
# ---------------------------------------------------------------------------

class TestBuildExpressionList:
    def test_fourier_method(self):
        """Fourier method: 3 exprs per contour."""
        coeffs = _make_coeffs(5)
        results = [{"coeffs": coeffs}, {"coeffs": coeffs}]
        exprs = build_desmos_expression_list(results, method="fourier", n_terms=3)
        assert len(exprs) == 6  # 3 per contour × 2 contours

    def test_piecewise_method(self):
        """Piecewise method: one expr per segment per contour."""
        segs = _make_segments(3)
        results = [{"segments": segs}, {"segments": segs}]
        exprs = build_desmos_expression_list(
            results, method="piecewise", max_segs_per_contour=3
        )
        assert len(exprs) == 6  # 3 segs × 2 contours

    def test_max_contours_respected(self):
        """max_contours should cap the number of processed contours."""
        coeffs = _make_coeffs(5)
        results = [{"coeffs": coeffs}] * 10
        exprs = build_desmos_expression_list(
            results, method="fourier", n_terms=3, max_contours=4
        )
        assert len(exprs) == 12  # 3 × 4 contours

    def test_skips_empty_results(self):
        """Results with no coeffs/segments should be silently skipped."""
        coeffs = _make_coeffs(5)
        results = [{"coeffs": coeffs}, {"coeffs": []}, {"coeffs": coeffs}]
        exprs = build_desmos_expression_list(results, method="fourier", n_terms=3)
        assert len(exprs) == 6  # empty middle contour skipped

    def test_all_ids_unique(self):
        """All IDs in the combined list must be unique."""
        coeffs = _make_coeffs(5)
        results = [{"coeffs": coeffs}] * 5
        exprs = build_desmos_expression_list(results, method="fourier", n_terms=3)
        ids = [e["id"] for e in exprs]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in expression list"


# ---------------------------------------------------------------------------
# expression_list_to_json
# ---------------------------------------------------------------------------

class TestExpressionListToJson:
    def test_valid_json(self):
        """Output must be valid JSON."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        json_str = expression_list_to_json(exprs)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_roundtrip(self):
        """Parsed JSON should match original expression list."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        json_str = expression_list_to_json(exprs)
        parsed = json.loads(json_str)
        assert len(parsed) == len(exprs)
        for orig, parsed_item in zip(exprs, parsed):
            assert orig["id"] == parsed_item["id"]
            assert orig["latex"] == parsed_item["latex"]
            assert orig["color"] == parsed_item["color"]


# ---------------------------------------------------------------------------
# expression_list_to_desmos_state  (directly loadable at desmos.com)
# ---------------------------------------------------------------------------

class TestExpressionListToDesmosState:
    def _get_state(self, n_contours: int = 1, n_terms: int = 2) -> dict:
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list(
            [{"coeffs": coeffs}] * n_contours, n_terms=n_terms
        )
        return json.loads(expression_list_to_desmos_state(exprs))

    def test_valid_json(self):
        """Output must be valid JSON."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        state_str = expression_list_to_desmos_state(exprs)
        state = json.loads(state_str)
        assert isinstance(state, dict)

    def test_top_level_keys(self):
        """State must have version, graph, and expressions keys."""
        state = self._get_state()
        assert "version" in state
        assert "graph" in state
        assert "expressions" in state

    def test_version_is_9(self):
        """Desmos state version should be 9."""
        state = self._get_state()
        assert state["version"] == 9

    def test_graph_has_viewport(self):
        """graph.viewport must be present with xmin/xmax/ymin/ymax."""
        state = self._get_state()
        vp = state["graph"]["viewport"]
        for key in ("xmin", "xmax", "ymin", "ymax"):
            assert key in vp, f"Missing viewport key: {key!r}"

    def test_expressions_list_present(self):
        """expressions.list must be a list."""
        state = self._get_state()
        assert isinstance(state["expressions"]["list"], list)

    def test_each_entry_has_type_expression(self):
        """Every entry in expressions.list must have type='expression'."""
        state = self._get_state()
        for entry in state["expressions"]["list"]:
            assert entry.get("type") == "expression", \
                f"Missing/wrong type on entry: {entry.get('id')!r}"

    def test_each_entry_has_required_fields(self):
        """Every entry must have id, color, and latex."""
        state = self._get_state()
        for entry in state["expressions"]["list"]:
            assert "id" in entry
            assert "color" in entry
            assert "latex" in entry

    def test_expression_count_matches(self):
        """Number of entries in state should match expression list length."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        state = json.loads(expression_list_to_desmos_state(exprs))
        assert len(state["expressions"]["list"]) == len(exprs)

    def test_hidden_flag_propagated(self):
        """Hidden flag from expression dicts must appear in state entries."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        hidden_exprs = [e for e in exprs if e.get("hidden")]
        assert len(hidden_exprs) > 0, "Expected some hidden helper expressions"
        state = json.loads(expression_list_to_desmos_state(exprs))
        state_ids = {e["id"]: e for e in state["expressions"]["list"]}
        for expr in hidden_exprs:
            assert state_ids[expr["id"]].get("hidden") is True

    def test_custom_viewport(self):
        """A custom viewport should appear in the state."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        vp = {"xmin": -100, "xmax": 100, "ymin": -50, "ymax": 50}
        state = json.loads(expression_list_to_desmos_state(exprs, viewport=vp))
        assert state["graph"]["viewport"] == vp


# ---------------------------------------------------------------------------
# build_desmos_html
# ---------------------------------------------------------------------------

class TestBuildDesmosHtml:
    def test_contains_api_script(self):
        """HTML must include the Desmos API script tag."""
        html = build_desmos_html([])
        assert "desmos.com/api" in html, "Missing Desmos API script tag"

    def test_contains_set_expression(self):
        """HTML must call setExpression for injecting equations."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        html = build_desmos_html(exprs)
        assert "setExpression" in html

    def test_contains_calculator_init(self):
        """HTML must initialise a GraphingCalculator instance."""
        html = build_desmos_html([])
        assert "GraphingCalculator" in html

    def test_expressions_serialised_in_html(self):
        """All expression IDs should appear somewhere in the HTML."""
        coeffs = _make_coeffs(3)
        exprs = build_desmos_expression_list([{"coeffs": coeffs}], n_terms=2)
        html = build_desmos_html(exprs)
        for expr in exprs:
            assert expr["id"] in html, f"ID {expr['id']!r} not found in HTML"

    def test_custom_title(self):
        """Custom title should appear in the <title> tag."""
        html = build_desmos_html([], title="My Custom Graph")
        assert "My Custom Graph" in html

    def test_valid_html_structure(self):
        """HTML must contain basic structural tags."""
        html = build_desmos_html([])
        for tag in ["<!DOCTYPE html>", "<html", "<head>", "<body>", "</html>"]:
            assert tag in html, f"Missing tag: {tag!r}"
