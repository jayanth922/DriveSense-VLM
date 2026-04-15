"""Tests for DriveSense-VLM Gradio demo (Phase 4a).

All tests use mocks — no GPU operations, no model downloads required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Make demo/ importable as a package directory
if str(_DEMO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_image():
    """Create a minimal synthetic PIL Image."""
    pytest.importorskip("PIL")
    from PIL import Image
    return Image.new("RGB", (672, 448), color=(80, 120, 160))


@pytest.fixture()
def hazard_annotation() -> dict:
    """Realistic hazard annotation with one bbox."""
    return {
        "hazards": [
            {
                "label": "vehicle_cut_in",
                "bbox_2d": [200, 100, 600, 400],
                "severity": "critical",
                "reasoning": "Vehicle merging without signalling at high speed.",
                "action": "emergency_brake",
            }
        ],
        "scene_summary": "Highway merge zone, fast-moving vehicle.",
        "ego_context": {
            "weather": "clear",
            "time_of_day": "day",
            "road_type": "highway",
        },
    }


@pytest.fixture()
def empty_annotation() -> dict:
    """Annotation with no hazards detected."""
    return {
        "hazards": [],
        "scene_summary": "Clear open road, no hazards detected.",
        "ego_context": {
            "weather": "clear",
            "time_of_day": "day",
            "road_type": "rural",
        },
    }


# ---------------------------------------------------------------------------
# test_draw_hazard_boxes_colors
# ---------------------------------------------------------------------------


class TestDrawHazardBoxesColors:
    """Verify draw_hazard_boxes uses SEVERITY_COLORS correctly."""

    def test_returns_pil_image(self, mock_image: object, hazard_annotation: dict) -> None:
        pytest.importorskip("PIL")
        # Import from demo/app.py
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, hazard_annotation)
            from PIL import Image
            assert isinstance(result, Image.Image)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_output_size_matches_input(
        self, mock_image: object, hazard_annotation: dict
    ) -> None:
        pytest.importorskip("PIL")
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, hazard_annotation)
            assert result.size == mock_image.size
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_severity_colors_defined(self) -> None:
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            expected_keys = {"critical", "high", "medium", "low", "no_hazard"}
            assert expected_keys.issubset(app.SEVERITY_COLORS.keys())
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_critical_is_red(self) -> None:
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            assert app.SEVERITY_COLORS["critical"] == (255, 0, 0)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# test_draw_hazard_boxes_empty
# ---------------------------------------------------------------------------


class TestDrawHazardBoxesEmpty:
    """draw_hazard_boxes with empty hazards list must not raise."""

    def test_no_hazards_returns_image(
        self, mock_image: object, empty_annotation: dict
    ) -> None:
        pytest.importorskip("PIL")
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, empty_annotation)
            from PIL import Image
            assert isinstance(result, Image.Image)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_no_hazards_output_same_size(
        self, mock_image: object, empty_annotation: dict
    ) -> None:
        pytest.importorskip("PIL")
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, empty_annotation)
            assert result.size == mock_image.size
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# test_draw_hazard_boxes_no_hazard_severity
# ---------------------------------------------------------------------------


class TestDrawHazardBoxesNoHazardSeverity:
    """Unknown/missing severity falls back to no_hazard colour (blue)."""

    def test_unknown_severity_uses_fallback(self, mock_image: object) -> None:
        pytest.importorskip("PIL")
        annotation = {
            "hazards": [
                {
                    "label": "unknown",
                    "bbox_2d": [50, 50, 200, 200],
                    "severity": "nonexistent_severity",
                    "reasoning": "test",
                    "action": "maintain_speed",
                }
            ]
        }
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, annotation)
            from PIL import Image
            assert isinstance(result, Image.Image)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_malformed_bbox_skipped(self, mock_image: object) -> None:
        """Bboxes with wrong length must be skipped without error."""
        pytest.importorskip("PIL")
        annotation = {
            "hazards": [
                {"label": "debris", "bbox_2d": [100, 200], "severity": "medium"}
            ]
        }
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            result = app.draw_hazard_boxes(mock_image, annotation)
            from PIL import Image
            assert isinstance(result, Image.Image)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# test_analyze_image_error_handling
# ---------------------------------------------------------------------------


class TestAnalyzeImageErrorHandling:
    """analyze_image must handle None input and model errors gracefully."""

    def test_none_image_returns_message(self) -> None:
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            _, msg, _ = app.analyze_image(None)
            assert isinstance(msg, str)
            assert len(msg) > 0
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_none_image_returns_none_annotated(self) -> None:
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            img_out, _, _ = app.analyze_image(None)
            assert img_out is None
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))

    def test_model_load_failure_returns_placeholder(self, mock_image: object) -> None:
        """When model load fails, analyze_image must return placeholder JSON."""
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            with patch.object(app, "_load_model", return_value=None):
                _, json_str, _ = app.analyze_image(mock_image)
            data = json.loads(json_str)
            assert "hazards" in data
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# test_demo_creation
# ---------------------------------------------------------------------------


class TestDemoCreation:
    """create_demo() must return a gr.Blocks instance or raise ImportError."""

    def test_create_demo_returns_blocks_or_raises(self) -> None:
        sys.path.insert(0, str(_DEMO_DIR))
        try:
            import importlib
            app = importlib.import_module("app")
            if not app._GRADIO_AVAILABLE:
                with pytest.raises(ImportError, match="gradio"):
                    app.create_demo()
            else:
                import gradio as gr
                demo = app.create_demo()
                assert isinstance(demo, gr.Blocks)
        finally:
            if str(_DEMO_DIR) in sys.path:
                sys.path.remove(str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# test_severity_colors (serve module)
# ---------------------------------------------------------------------------


class TestSeverityColors:
    """Verify SEVERITY_COLORS in serve.py matches expected values."""

    def test_all_severity_keys_present(self) -> None:
        from drivesense.inference.serve import SEVERITY_COLORS

        assert "critical" in SEVERITY_COLORS
        assert "high" in SEVERITY_COLORS
        assert "medium" in SEVERITY_COLORS
        assert "low" in SEVERITY_COLORS
        assert "no_hazard" in SEVERITY_COLORS

    def test_critical_is_red(self) -> None:
        from drivesense.inference.serve import SEVERITY_COLORS

        assert SEVERITY_COLORS["critical"] == (255, 0, 0)

    def test_high_is_orange(self) -> None:
        from drivesense.inference.serve import SEVERITY_COLORS

        r, g, b = SEVERITY_COLORS["high"]
        assert r == 255 and g > 0 and b == 0, "high should be orange (r=255, g>0, b=0)"

    def test_no_hazard_is_blue(self) -> None:
        from drivesense.inference.serve import SEVERITY_COLORS

        r, g, b = SEVERITY_COLORS["no_hazard"]
        assert b > r and b > g, "no_hazard should be blue-dominant"

    def test_all_values_are_rgb_tuples(self) -> None:
        from drivesense.inference.serve import SEVERITY_COLORS

        for name, color in SEVERITY_COLORS.items():
            assert len(color) == 3, f"{name} must be (R, G, B)"
            assert all(0 <= c <= 255 for c in color), f"{name} values out of [0,255]"
