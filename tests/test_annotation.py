"""Tests for Phase 1c: LLM annotation pipeline.

All tests use MockLLMClient — no API calls are made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.annotation import (
    AnnotationPromptBuilder,
    AnnotationValidator,
    LLMAnnotationPipeline,
    MockLLMClient,
    SFTDataFormatter,
    _fill_scenario,
    _build_source_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUSCENES_FRAME: dict = {
    "frame_id": "nuscenes_abc123",
    "source": "nuscenes",
    "image_path": "/fake/path/frame.png",
    "scene_description": "Urban street with pedestrians at intersection",
    "weather": "clear",
    "time_of_day": "day",
    "road_type": "urban",
    "split": "train",
    "source_metadata": json.dumps({
        "rarity_signals": ["proximity", "occlusion"],
        "rarity_score": 4,
        "annotations": [
            {"category_name": "human.pedestrian.adult"},
            {"category_name": "vehicle.car"},
        ],
    }),
}

_DADA_FRAME: dict = {
    "frame_id": "dada_cat01_seq001_frame0030_critical",
    "source": "dada2000",
    "image_path": "/fake/path/dada_frame.png",
    "scene_description": "Highway accident scenario with vehicle collision",
    "weather": "rain",
    "time_of_day": "night",
    "road_type": "highway",
    "split": "train",
    "category": "001",
    "sequence": "001",
    "frame_type": "critical",
}

_VALID_ANNOTATION: dict = {
    "hazards": [
        {
            "bbox_2d": [100, 200, 400, 600],
            "label": "occluded_pedestrian",
            "severity": "high",
            "reasoning": (
                "Adult pedestrian stepping off curb partially occluded by parked van. "
                "Ego vehicle has limited reaction distance at current speed."
            ),
            "action": "Decelerate immediately and prepare to yield.",
        }
    ],
    "scene_summary": "Urban intersection with moderate traffic and one occluded pedestrian.",
    "ego_context": {"weather": "clear", "time_of_day": "day", "road_type": "urban"},
}

_MINIMAL_CONFIG: dict = {
    "annotation": {
        "llm_model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 1024,
        "retry_attempts": 1,
        "retry_backoff_base": 1.0,
        "cache_dir": "",  # will be overridden per test
        "output_dir": "",
        "sft_output_dir": "",
        "counterfactual_ratio": 0.3,
        "max_concurrent_requests": 1,
    }
}


@pytest.fixture()
def builder() -> AnnotationPromptBuilder:
    return AnnotationPromptBuilder()


@pytest.fixture()
def validator() -> AnnotationValidator:
    return AnnotationValidator()


@pytest.fixture()
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture()
def pipeline(tmp_path: Path) -> LLMAnnotationPipeline:
    cfg = dict(_MINIMAL_CONFIG)
    cfg["annotation"] = dict(cfg["annotation"])
    cfg["annotation"]["cache_dir"] = str(tmp_path / "cache")
    return LLMAnnotationPipeline(cfg, mock_client=MockLLMClient())


# ---------------------------------------------------------------------------
# AnnotationPromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_prompt_builder_real_nuscenes(self, builder: AnnotationPromptBuilder) -> None:
        """Real annotation prompt contains nuScenes rarity signals."""
        system, user = builder.build_annotation_prompt(_NUSCENES_FRAME)
        assert "rarity" in user.lower() or "proximity" in user.lower()
        assert "nuscenes" in user.lower()
        assert len(system) > 50
        assert "{" not in user or "output_schema" not in user  # placeholders resolved

    def test_prompt_builder_real_dada(self, builder: AnnotationPromptBuilder) -> None:
        """Real annotation prompt contains DADA-2000 accident context."""
        system, user = builder.build_annotation_prompt(_DADA_FRAME)
        assert "dada2000" in user.lower()
        assert "critical" in user.lower()
        assert len(system) > 50

    def test_prompt_builder_source_specific_context_nuscenes(
        self, builder: AnnotationPromptBuilder
    ) -> None:
        """NuScenes prompt includes agent categories from annotations."""
        system, user = builder.build_annotation_prompt(_NUSCENES_FRAME)
        assert "human.pedestrian" in user or "pedestrian" in user.lower()

    def test_prompt_builder_counterfactual(self, builder: AnnotationPromptBuilder) -> None:
        """Counterfactual prompt includes the scenario text."""
        system, user, meta = builder.build_counterfactual_prompt(_NUSCENES_FRAME)
        assert meta["scenario_label"] in AnnotationValidator.VALID_LABELS
        assert meta["scenario_text"] in user
        assert len(system) > 50

    def test_prompt_builder_counterfactual_scenario_has_text(
        self, builder: AnnotationPromptBuilder
    ) -> None:
        """Counterfactual prompt scenario text is non-empty."""
        _, _, meta = builder.build_counterfactual_prompt(_DADA_FRAME)
        assert len(meta["scenario_text"]) > 10

    def test_prompt_builder_output_schema_injected(self, builder: AnnotationPromptBuilder) -> None:
        """Output schema is embedded in user prompt."""
        _, user = builder.build_annotation_prompt(_NUSCENES_FRAME)
        assert "bbox_2d" in user
        assert "hazards" in user

    def test_get_output_schema(self, builder: AnnotationPromptBuilder) -> None:
        """get_output_schema returns a non-empty string with required keys."""
        schema = builder.get_output_schema()
        assert "hazards" in schema
        assert "bbox_2d" in schema
        assert "scene_summary" in schema
        assert "ego_context" in schema

    def test_fill_scenario_replaces_variables(self) -> None:
        """_fill_scenario fills all template variables."""
        tpl = {
            "scenario": "A {side} pedestrian {distance} away.",
            "label": "jaywalking",
            "variables": {"side": ["left"], "distance": ["20 meters"]},
        }
        result = _fill_scenario(tpl)
        assert "{side}" not in result
        assert "{distance}" not in result
        assert "left" in result
        assert "20 meters" in result

    def test_build_source_context_empty_for_unknown(self) -> None:
        """Unknown source produces empty context string."""
        ctx = _build_source_context("unknown_source", {}, {})
        assert ctx == ""


# ---------------------------------------------------------------------------
# AnnotationValidator tests
# ---------------------------------------------------------------------------

class TestAnnotationValidator:
    def test_valid_annotation_passes(self, validator: AnnotationValidator) -> None:
        """A well-formed annotation passes validation."""
        is_valid, errors = validator.validate_annotation(_VALID_ANNOTATION)
        assert is_valid, errors

    def test_invalid_bbox_range(self, validator: AnnotationValidator) -> None:
        """Bbox value outside [0, 1000] is caught."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["bbox_2d"] = [100, 200, 400, 1500]
        is_valid, errors = validator.validate_annotation(bad)
        assert not is_valid
        assert any("1500" in e or "out of" in e for e in errors)

    def test_invalid_label(self, validator: AnnotationValidator) -> None:
        """Unknown label is caught."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["label"] = "giant_spider"
        is_valid, errors = validator.validate_annotation(bad)
        assert not is_valid
        assert any("label" in e for e in errors)

    def test_inverted_bbox_caught(self, validator: AnnotationValidator) -> None:
        """x1 >= x2 is caught."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["bbox_2d"] = [500, 200, 100, 600]  # x1 > x2
        is_valid, errors = validator.validate_annotation(bad)
        assert not is_valid
        assert any("x1" in e for e in errors)

    def test_missing_fields(self, validator: AnnotationValidator) -> None:
        """Missing top-level keys are reported."""
        is_valid, errors = validator.validate_annotation({"hazards": []})
        assert not is_valid
        assert any("scene_summary" in e or "ego_context" in e for e in errors)

    def test_short_reasoning_caught(self, validator: AnnotationValidator) -> None:
        """Reasoning shorter than 20 chars is flagged."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["reasoning"] = "Too short."
        is_valid, errors = validator.validate_annotation(bad)
        assert not is_valid
        assert any("reasoning" in e for e in errors)

    def test_fix_clamped_coords(self, validator: AnnotationValidator) -> None:
        """Coords > 1000 are clamped to 1000 by fix_common_issues."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["bbox_2d"] = [0, 0, 1200, 1500]
        fixed = validator.fix_common_issues(bad)
        bbox = fixed["hazards"][0]["bbox_2d"]
        assert all(0 <= v <= 1000 for v in bbox)

    def test_fix_inverted_bbox(self, validator: AnnotationValidator) -> None:
        """Inverted bbox is corrected by fix_common_issues."""
        bad = json.loads(json.dumps(_VALID_ANNOTATION))
        bad["hazards"][0]["bbox_2d"] = [600, 700, 100, 200]  # x1>x2, y1>y2
        fixed = validator.fix_common_issues(bad)
        x1, y1, x2, y2 = fixed["hazards"][0]["bbox_2d"]
        assert x1 < x2
        assert y1 < y2

    def test_fix_adds_ego_context_defaults(self, validator: AnnotationValidator) -> None:
        """Missing ego_context is inserted with defaults."""
        bad = {"hazards": [_VALID_ANNOTATION["hazards"][0]], "scene_summary": "test"}
        fixed = validator.fix_common_issues(bad)
        assert "ego_context" in fixed
        assert fixed["ego_context"]["weather"] == "clear"

    def test_parse_json_with_code_fences(self, validator: AnnotationValidator) -> None:
        """JSON inside code fences is extracted correctly."""
        text = '```json\n{"hazards": [], "scene_summary": "x", "ego_context": {}}\n```'
        result = validator.parse_llm_response(text)
        assert isinstance(result, dict)
        assert "hazards" in result

    def test_parse_json_with_surrounding_text(self, validator: AnnotationValidator) -> None:
        """JSON embedded in prose is extracted."""
        text = 'Here is my analysis:\n{"hazards": [], "scene_summary": "s", "ego_context": {}}\nDone.'
        result = validator.parse_llm_response(text)
        assert isinstance(result, dict)

    def test_parse_invalid_json_returns_none(self, validator: AnnotationValidator) -> None:
        """Unparseable text returns None."""
        result = validator.parse_llm_response("This is not JSON at all.")
        assert result is None


# ---------------------------------------------------------------------------
# SFTDataFormatter tests
# ---------------------------------------------------------------------------

class TestSFTDataFormatter:
    def test_format_single_example_structure(self) -> None:
        """SFT example has correct message structure."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION}
        example = formatter.format_single_example(frame)
        assert "messages" in example
        assert "images" in example
        assert "frame_id" in example
        assert "source" in example
        assert len(example["messages"]) == 3

    def test_format_has_system_prompt(self) -> None:
        """SFT example first message is a system prompt."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION}
        example = formatter.format_single_example(frame)
        assert example["messages"][0]["role"] == "system"
        assert "DriveSense" in example["messages"][0]["content"]

    def test_format_image_path_in_user(self) -> None:
        """SFT user message content includes image path."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION}
        example = formatter.format_single_example(frame)
        user_content = example["messages"][1]["content"]
        assert any(item.get("type") == "image" for item in user_content)

    def test_format_assistant_is_json_string(self) -> None:
        """SFT assistant turn is a JSON string of the annotation."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION}
        example = formatter.format_single_example(frame)
        assistant_content = example["messages"][2]["content"]
        parsed = json.loads(assistant_content)
        assert "hazards" in parsed

    def test_validate_sft_example_valid(self) -> None:
        """Well-formed SFT example passes validate_sft_example."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION}
        example = formatter.format_single_example(frame)
        assert formatter.validate_sft_example(example)

    def test_validate_sft_example_missing_images(self) -> None:
        """SFT example with empty images fails validation."""
        formatter = SFTDataFormatter()
        frame = {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION, "image_path": ""}
        example = formatter.format_single_example(frame)
        example["images"] = []
        assert not formatter.validate_sft_example(example)

    def test_format_dataset_writes_splits(self, tmp_path: Path) -> None:
        """format_dataset creates sft_train.jsonl, sft_val.jsonl, sft_test.jsonl."""
        formatter = SFTDataFormatter()
        frames = [
            {**_NUSCENES_FRAME, "annotations": _VALID_ANNOTATION, "split": "train"},
            {**_DADA_FRAME, "annotations": _VALID_ANNOTATION, "split": "val"},
        ]
        manifest_path = tmp_path / "annotated_manifest.json"
        manifest_path.write_text(json.dumps(frames), encoding="utf-8")

        out = formatter.format_dataset(manifest_path, tmp_path / "sft")
        assert (out / "sft_train.jsonl").exists()
        assert (out / "sft_val.jsonl").exists()
        assert (out / "sft_format_stats.json").exists()


# ---------------------------------------------------------------------------
# LLMAnnotationPipeline tests (mock LLM)
# ---------------------------------------------------------------------------

class TestAnnotationPipeline:
    def test_annotation_caching(self, pipeline: LLMAnnotationPipeline) -> None:
        """Second call for same frame returns cached result without calling LLM."""
        result1 = pipeline.annotate_frame(_NUSCENES_FRAME)
        assert result1 is not None

        # Replace mock with one that fails if called
        pipeline._mock = MagicMock(side_effect=RuntimeError("should not call"))
        result2 = pipeline.annotate_frame(_NUSCENES_FRAME)
        assert result2 is not None
        assert result2["frame_id"] == result1["frame_id"]

    def test_annotate_frame_returns_valid_structure(
        self, pipeline: LLMAnnotationPipeline
    ) -> None:
        """annotate_frame returns dict with expected keys."""
        result = pipeline.annotate_frame(_NUSCENES_FRAME)
        assert result is not None
        assert "frame_id" in result
        assert "annotation" in result
        assert "mode" in result

    def test_annotate_frame_dada_source(self, pipeline: LLMAnnotationPipeline) -> None:
        """annotate_frame works for DADA-2000 frames."""
        result = pipeline.annotate_frame(_DADA_FRAME)
        assert result is not None
        ann = result["annotation"]
        assert "hazards" in ann
        assert ann["hazards"][0]["label"] == "occluded_pedestrian"

    def test_mock_llm_pipeline_end_to_end(self, tmp_path: Path) -> None:
        """run_full_pipeline with mock LLM produces all expected output files."""
        cfg = dict(_MINIMAL_CONFIG)
        cfg["annotation"] = dict(cfg["annotation"])
        cfg["annotation"]["cache_dir"] = str(tmp_path / "cache")

        frames = [
            {**_NUSCENES_FRAME, "frame_id": "ns_001", "split": "train"},
            {**_NUSCENES_FRAME, "frame_id": "ns_002", "split": "val"},
            {**_DADA_FRAME, "frame_id": "da_001", "split": "train"},
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            "\n".join(json.dumps(f) for f in frames), encoding="utf-8"
        )

        pipeline = LLMAnnotationPipeline(cfg, mock_client=MockLLMClient())
        out_dir = pipeline.run_full_pipeline(
            manifest_path=manifest_path,
            output_dir=tmp_path / "annotated",
            counterfactual_ratio=0.0,  # skip CF for speed
        )

        assert (out_dir / "annotated_manifest.json").exists()
        assert (out_dir / "annotated_manifest_train.json").exists()
        assert (out_dir / "annotated_manifest_val.json").exists()
        assert (out_dir / "quality_report.json").exists()
        assert (out_dir / "failed_annotations.json").exists()

    def test_quality_report_has_all_fields(self, pipeline: LLMAnnotationPipeline) -> None:
        """generate_quality_report returns all expected metric keys."""
        result = pipeline.annotate_frame(_NUSCENES_FRAME)
        report = pipeline.generate_quality_report([result])
        for key in (
            "total_frames",
            "success_count",
            "success_rate",
            "auto_fix_count",
            "label_distribution",
            "severity_distribution",
            "avg_hazards_per_frame",
            "avg_reasoning_length_chars",
            "real_annotation_count",
            "counterfactual_count",
        ):
            assert key in report, f"Missing key: {key}"

    def test_counterfactual_mode(self, pipeline: LLMAnnotationPipeline) -> None:
        """annotate_frame in counterfactual mode includes scenario_metadata and is_counterfactual."""
        result = pipeline.annotate_frame(_NUSCENES_FRAME, mode="counterfactual")
        assert result is not None
        assert result.get("mode") == "counterfactual"
        assert result.get("is_counterfactual") is True
        assert "scenario_metadata" in result

    def test_real_mode_not_flagged_as_counterfactual(
        self, pipeline: LLMAnnotationPipeline
    ) -> None:
        """Real annotations have is_counterfactual=False."""
        result = pipeline.annotate_frame(_NUSCENES_FRAME, mode="real")
        assert result is not None
        assert result.get("is_counterfactual") is False

    def test_counterfactual_scenario_label_valid(self, pipeline: LLMAnnotationPipeline) -> None:
        """Counterfactual scenario_label is one of the valid hazard labels."""
        result = pipeline.annotate_frame(_NUSCENES_FRAME, mode="counterfactual")
        assert result is not None
        label = result["scenario_metadata"]["scenario_label"]
        assert label in AnnotationValidator.VALID_LABELS
