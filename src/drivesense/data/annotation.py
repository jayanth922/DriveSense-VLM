"""LLM-based annotation and counterfactual augmentation pipeline.

Implements Phase 1c: call an LLM (Anthropic Claude) with vision to generate
structured JSON hazard annotations for all unified dataset frames.  For a
configurable fraction of frames, also generates counterfactual augmentations
by prompting the model with a hypothetical hazard scenario.

Target SFT annotation schema:
    {
        "hazards": [
            {
                "bbox_2d": [x1, y1, x2, y2],   # integers in [0, 1000]
                "label": str,                    # one of VALID_LABELS
                "severity": str,                 # one of VALID_SEVERITIES
                "reasoning": str,
                "action": str
            }
        ],
        "scene_summary": str,
        "ego_context": {"weather": str, "time_of_day": str, "road_type": str}
    }

Phase 1c implementation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from anthropic import Anthropic  # type: ignore[import]
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_OUTPUT_SCHEMA = """{
    "hazards": [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "<one of: occluded_pedestrian | jaywalking | cyclist_proximity"
            " | construction_zone | adverse_weather | unusual_object"
            " | emergency_vehicle | high_density | no_hazard>",
            "severity": "<critical | high | medium | low>",
            "reasoning": "<2-3 sentences explaining why this is dangerous>",
            "action": "<specific driving action to take>"
        }
    ],
    "scene_summary": "<1-2 sentence description of the overall scene>",
    "ego_context": {
        "weather": "<clear | rain | fog | snow | night>",
        "time_of_day": "<day | night | dawn | dusk>",
        "road_type": "<urban | highway | intersection | parking>"
    }
}"""

# ---------------------------------------------------------------------------
# AnnotationPromptBuilder
# ---------------------------------------------------------------------------


class AnnotationPromptBuilder:
    """Builds annotation prompts from disk templates and frame metadata.

    Templates live in ``src/drivesense/data/prompts/`` as plain text files
    so they can be version-controlled and edited without touching Python code.

    Args:
        prompts_dir: Override for the prompts directory (default: next to this file).
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
        self._system_tpl = self._load("annotation_system.txt")
        self._user_tpl = self._load("annotation_user.txt")
        self._cf_user_tpl = self._load("counterfactual_user.txt")
        self._scenarios: list[dict] = self._load_scenarios()

    def _load(self, filename: str) -> str:
        """Read a prompt template file."""
        path = self._dir / filename
        return path.read_text(encoding="utf-8").strip()

    def _load_scenarios(self) -> list[dict]:
        """Load the counterfactual scenario bank."""
        path = self._dir / "counterfactual_scenarios.json"
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def get_output_schema(self) -> str:
        """Return the JSON output schema string for prompt injection."""
        return _OUTPUT_SCHEMA

    def build_annotation_prompt(self, frame: dict) -> tuple[str, str]:
        """Build system + user prompt for real annotation.

        Populates the user template with frame metadata and adds
        source-specific context (rarity signals for nuScenes, accident
        category for DADA-2000).

        Args:
            frame: Unified frame dict from the manifest.

        Returns:
            ``(system_prompt, user_prompt)``
        """
        source = frame.get("source", "unknown")
        source_meta = frame.get("source_metadata", {})
        if isinstance(source_meta, str):
            try:
                source_meta = json.loads(source_meta)
            except (json.JSONDecodeError, TypeError):
                source_meta = {}

        source_ctx = _build_source_context(source, source_meta, frame)

        user = self._user_tpl.format(
            source=source,
            scene_description=frame.get("scene_description") or frame.get("description") or "N/A",
            weather=frame.get("weather", "unknown"),
            time_of_day=frame.get("time_of_day", "unknown"),
            road_type=frame.get("road_type", "unknown"),
            source_specific_context=source_ctx,
            output_schema=self.get_output_schema(),
        )
        return self._system_tpl, user

    def build_counterfactual_prompt(self, frame: dict) -> tuple[str, str, dict]:
        """Build system + user prompt for counterfactual augmentation.

        Randomly selects a scenario from the bank that is appropriate for
        the frame's road type, fills template variables, and returns the
        completed prompts plus the scenario metadata.

        Args:
            frame: Unified frame dict from the manifest.

        Returns:
            ``(system_prompt, user_prompt, scenario_metadata)``
        """
        road_type = frame.get("road_type", "urban")
        eligible = [
            s for s in self._scenarios
            if road_type not in s.get("exclude_road_types", [])
        ]
        if not eligible:
            eligible = self._scenarios  # fallback: use all

        scenario_tpl = random.choice(eligible)  # noqa: S311
        filled = _fill_scenario(scenario_tpl)

        user = self._cf_user_tpl.format(
            scene_description=frame.get("scene_description") or frame.get("description") or "N/A",
            weather=frame.get("weather", "unknown"),
            time_of_day=frame.get("time_of_day", "unknown"),
            road_type=road_type,
            counterfactual_scenario=filled,
            output_schema=self.get_output_schema(),
        )
        meta = {"scenario_label": scenario_tpl["label"], "scenario_text": filled}
        return self._system_tpl, user, meta


def _build_source_context(source: str, source_meta: dict, frame: dict) -> str:
    """Build the source-specific context block for the annotation prompt.

    Args:
        source: Frame source identifier (``"nuscenes"`` or ``"dada2000"``).
        source_meta: Parsed ``source_metadata`` field from the frame.
        frame: Full frame dict for fallback lookups.

    Returns:
        Formatted context string to embed in the prompt.
    """
    lines: list[str] = []
    if source == "nuscenes":
        signals = source_meta.get("rarity_signals", [])
        score = source_meta.get("rarity_score", source_meta.get("composite_score"))
        if signals:
            lines.append(f"- Rarity signals: {', '.join(signals)}")
        if score is not None:
            lines.append(f"- Rarity score: {score}/6")
        anns = source_meta.get("annotations") or frame.get("annotations", [])
        if isinstance(anns, list) and anns:
            cats = {a.get("category_name", "") for a in anns if isinstance(a, dict)}
            cats.discard("")
            if cats:
                lines.append(f"- Agent categories: {', '.join(sorted(cats))}")
    elif source == "dada2000":
        cat = frame.get("category") or source_meta.get("category")
        seq = frame.get("sequence") or source_meta.get("sequence")
        ftype = frame.get("frame_type") or source_meta.get("frame_type")
        if cat:
            lines.append(f"- Accident category: {cat}")
        if seq:
            lines.append(f"- Sequence: {seq}")
        if ftype:
            lines.append(f"- Frame type: {ftype}")
    return ("\n" + "\n".join(lines)) if lines else ""


def _fill_scenario(scenario_tpl: dict) -> str:
    """Fill template variables in a counterfactual scenario.

    Args:
        scenario_tpl: Scenario dict with ``scenario`` string and ``variables`` map.

    Returns:
        Scenario string with all ``{variable}`` placeholders replaced.
    """
    text = scenario_tpl["scenario"]
    for var, choices in scenario_tpl.get("variables", {}).items():
        if choices:
            text = text.replace("{" + var + "}", random.choice(choices))  # noqa: S311
    return text


# ---------------------------------------------------------------------------
# AnnotationValidator
# ---------------------------------------------------------------------------


class AnnotationValidator:
    """Validates LLM-generated annotations against the expected schema.

    Catches malformed JSON, out-of-range coordinates, invalid labels,
    missing fields, and other common LLM output issues.
    """

    VALID_LABELS: list[str] = [
        "occluded_pedestrian",
        "jaywalking",
        "cyclist_proximity",
        "construction_zone",
        "adverse_weather",
        "unusual_object",
        "emergency_vehicle",
        "high_density",
        "no_hazard",
    ]
    VALID_SEVERITIES: list[str] = ["critical", "high", "medium", "low"]

    @staticmethod
    def validate_annotation(annotation: dict) -> tuple[bool, list[str]]:
        """Validate a single annotation dict against the target schema.

        Args:
            annotation: Parsed annotation dict.

        Returns:
            ``(is_valid, error_messages)`` — valid iff error_messages is empty.
        """
        errors: list[str] = []
        for top_key in ("hazards", "scene_summary", "ego_context"):
            if top_key not in annotation:
                errors.append(f"Missing top-level key: '{top_key}'")

        hazards = annotation.get("hazards")
        if not isinstance(hazards, list):
            errors.append("'hazards' must be a list")
        elif len(hazards) == 0:
            errors.append("'hazards' list is empty")
        else:
            for i, h in enumerate(hazards):
                errors.extend(_validate_hazard(h, i))

        ego = annotation.get("ego_context")
        if isinstance(ego, dict):
            for k in ("weather", "time_of_day", "road_type"):
                if k not in ego:
                    errors.append(f"ego_context missing key: '{k}'")
        elif ego is not None:
            errors.append("'ego_context' must be a dict")

        if not annotation.get("scene_summary"):
            errors.append("'scene_summary' is empty or missing")

        return (len(errors) == 0), errors

    @staticmethod
    def fix_common_issues(annotation: dict) -> dict:
        """Attempt to repair common LLM annotation issues in-place.

        Fixes applied:
        - Clamp bbox coords to [0, 1000]
        - Swap inverted x1/x2 or y1/y2
        - Normalise label casing
        - Add missing ``ego_context`` defaults

        Args:
            annotation: Annotation dict to fix (modified in-place copy).

        Returns:
            Fixed annotation dict.
        """
        annotation = dict(annotation)  # shallow copy
        hazards = annotation.get("hazards")
        if isinstance(hazards, list):
            annotation["hazards"] = [_fix_hazard(h) for h in hazards]
        if "ego_context" not in annotation or not isinstance(annotation["ego_context"], dict):
            annotation["ego_context"] = {
                "weather": "clear", "time_of_day": "day", "road_type": "urban"
            }
        else:
            ctx = annotation["ego_context"]
            ctx.setdefault("weather", "clear")
            ctx.setdefault("time_of_day", "day")
            ctx.setdefault("road_type", "urban")
        annotation.setdefault("scene_summary", "Scene summary unavailable.")
        return annotation

    @staticmethod
    def parse_llm_response(response_text: str) -> dict | None:
        """Parse an LLM response string into an annotation dict.

        Handles raw JSON, JSON inside ```json ... ``` code fences,
        and JSON embedded within surrounding prose.

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            Parsed dict or ``None`` if no valid JSON can be extracted.
        """
        # Strip code fences.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1)
            result = _try_parse_json(candidate)
            if result is not None:
                return result

        # Try the whole text as JSON.
        result = _try_parse_json(response_text.strip())
        if result is not None:
            return result

        # Find the first {...} block in mixed text.
        brace_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if brace_match:
            return _try_parse_json(brace_match.group(0))
        return None


def _validate_hazard(h: Any, idx: int) -> list[str]:  # noqa: ANN401
    """Validate a single hazard entry; return list of error strings."""
    errors: list[str] = []
    if not isinstance(h, dict):
        errors.append(f"hazards[{idx}] is not a dict")
        return errors
    for k in ("bbox_2d", "label", "severity", "reasoning", "action"):
        if k not in h:
            errors.append(f"hazards[{idx}] missing key '{k}'")
    bbox = h.get("bbox_2d")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"hazards[{idx}].bbox_2d must be a list of 4 integers")
        else:
            for v in bbox:
                if not isinstance(v, (int, float)) or v < 0 or v > 1000:
                    errors.append(f"hazards[{idx}].bbox_2d value {v} out of [0, 1000]")
                    break
            x1, y1, x2, y2 = bbox
            if x1 >= x2:
                errors.append(f"hazards[{idx}].bbox_2d x1 ({x1}) >= x2 ({x2})")
            if y1 >= y2:
                errors.append(f"hazards[{idx}].bbox_2d y1 ({y1}) >= y2 ({y2})")
    label = h.get("label", "")
    if label not in AnnotationValidator.VALID_LABELS:
        errors.append(f"hazards[{idx}].label '{label}' not in VALID_LABELS")
    severity = h.get("severity", "")
    if severity not in AnnotationValidator.VALID_SEVERITIES:
        errors.append(f"hazards[{idx}].severity '{severity}' not in VALID_SEVERITIES")
    reasoning = h.get("reasoning", "")
    if not isinstance(reasoning, str) or len(reasoning) < 20:
        errors.append(f"hazards[{idx}].reasoning too short (< 20 chars)")
    if not h.get("action"):
        errors.append(f"hazards[{idx}].action is empty")
    return errors


def _fix_hazard(h: Any) -> Any:  # noqa: ANN401
    """Attempt to repair a single hazard dict."""
    if not isinstance(h, dict):
        return h
    h = dict(h)
    # Normalise label casing.
    label = str(h.get("label", "")).lower().strip()
    if label in AnnotationValidator.VALID_LABELS:
        h["label"] = label
    # Clamp and un-invert bbox.
    bbox = h.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        clamped = [max(0, min(1000, int(round(v)))) for v in bbox]
        x1, y1, x2, y2 = clamped
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if x1 == x2:
            x2 = min(1000, x2 + 1)
        if y1 == y2:
            y2 = min(1000, y2 + 1)
        h["bbox_2d"] = [x1, y1, x2, y2]
    return h


def _try_parse_json(text: str) -> dict | None:
    """Return parsed dict if text is valid JSON, else None."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Returns template annotations for testing without API access.

    Produces deterministic, schema-valid annotations based on the frame source
    so tests can exercise the full pipeline without network calls.
    """

    def annotate(self, frame: dict, system: str, user: str) -> str:  # noqa: ARG002
        """Return a valid JSON annotation string for the given frame.

        Args:
            frame: Unified frame metadata dict.
            system: System prompt (unused in mock).
            user: User prompt (unused in mock).

        Returns:
            JSON string matching the target SFT annotation schema.
        """
        source = frame.get("source", "nuscenes")
        if source == "dada2000":
            hazard = {
                "bbox_2d": [300, 400, 600, 800],
                "label": "occluded_pedestrian",
                "severity": "high",
                "reasoning": (
                    "Pedestrian detected partially obscured at the scene of the annotated "
                    "accident event. Ego vehicle has limited reaction distance given the "
                    "current trajectory and speed."
                ),
                "action": "Decelerate immediately and prepare to stop.",
            }
            weather = frame.get("weather", "clear")
            time_of_day = frame.get("time_of_day", "day")
            road_type = frame.get("road_type", "urban")
        else:
            hazard = {
                "bbox_2d": [200, 300, 500, 750],
                "label": "high_density",
                "severity": "medium",
                "reasoning": (
                    "Scene contains multiple agents at elevated density based on rarity "
                    "signals. The high agent count increases unpredictability and requires "
                    "heightened attention from the ego vehicle."
                ),
                "action": "Reduce speed and increase following distance.",
            }
            weather = frame.get("weather", "clear")
            time_of_day = frame.get("time_of_day", "day")
            road_type = frame.get("road_type", "urban")

        annotation = {
            "hazards": [hazard],
            "scene_summary": (
                "Mock scene summary generated for testing purposes without API access."
            ),
            "ego_context": {
                "weather": weather,
                "time_of_day": time_of_day,
                "road_type": road_type,
            },
        }
        return json.dumps(annotation)


# ---------------------------------------------------------------------------
# LLMAnnotationPipeline
# ---------------------------------------------------------------------------


class LLMAnnotationPipeline:
    """End-to-end LLM-powered annotation pipeline with caching and async batching.

    Processes unified dataset frames through Claude (vision API) to generate
    structured hazard annotations for SFT training.  Supports:
    - File-based cache (one JSON per frame) for checkpoint/resume
    - Exponential-backoff retry on API errors
    - Async concurrent batch processing
    - Counterfactual augmentation
    - Quality report generation

    Args:
        config: Annotation config from ``configs/data.yaml``.
        api_key: Anthropic API key (falls back to ``ANTHROPIC_API_KEY`` env var).
        mock_client: If provided, use this instead of the real Anthropic client.
    """

    def __init__(
        self,
        config: dict,
        api_key: str | None = None,
        mock_client: MockLLMClient | None = None,
    ) -> None:
        ann_cfg = config.get("annotation", {})
        self._model: str = ann_cfg.get("llm_model", "claude-sonnet-4-20250514")
        self._temperature: float = float(ann_cfg.get("temperature", 0.3))
        self._max_tokens: int = int(ann_cfg.get("max_tokens", 1024))
        self._retry_attempts: int = int(ann_cfg.get("retry_attempts", 3))
        self._retry_backoff: float = float(ann_cfg.get("retry_backoff_base", 2.0))
        cache_dir = Path(ann_cfg.get("cache_dir", "outputs/data/annotation_cache"))
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._prompt_builder = AnnotationPromptBuilder()
        self._validator = AnnotationValidator()
        self._mock = mock_client

        if mock_client is None:
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic>=0.30"
                )
            self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, frame_id: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", frame_id)
        return self._cache_dir / f"{safe}.json"

    def _load_from_cache(self, frame_id: str) -> dict | None:
        p = self._cache_path(frame_id)
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _save_to_cache(self, frame_id: str, result: dict) -> None:
        p = self._cache_path(frame_id)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    # ------------------------------------------------------------------
    # Single-frame annotation
    # ------------------------------------------------------------------

    def annotate_frame(self, frame: dict, mode: str = "real") -> dict | None:
        """Annotate a single frame, using cache if available.

        Args:
            frame: Unified frame metadata dict.
            mode: ``"real"`` for standard annotation, ``"counterfactual"`` for augmentation.

        Returns:
            Annotation result dict (``frame_id`` + ``annotation`` + optional
            ``scenario_metadata``) or ``None`` if all retries fail.
        """
        frame_id = frame.get("frame_id", "unknown")
        cache_key = f"{frame_id}__{mode}"
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        scenario_meta: dict = {}
        if mode == "counterfactual":
            system, user, scenario_meta = self._prompt_builder.build_counterfactual_prompt(frame)
        else:
            system, user = self._prompt_builder.build_annotation_prompt(frame)

        for attempt in range(self._retry_attempts):
            try:
                response_text = self._call_llm(frame, system, user)
                annotation = self._validator.parse_llm_response(response_text)
                if annotation is None:
                    raise ValueError("Could not parse LLM response as JSON")
                is_valid, errors = self._validator.validate_annotation(annotation)
                if not is_valid:
                    annotation = self._validator.fix_common_issues(annotation)
                    is_valid, errors = self._validator.validate_annotation(annotation)
                if not is_valid:
                    logger.warning("Frame %s annotation invalid after fix: %s", frame_id, errors)
                    # Proceed anyway — flag in result
                result: dict = {
                    "frame_id": frame_id,
                    "mode": mode,
                    "is_counterfactual": mode == "counterfactual",
                    "annotation": annotation,
                    "validation_errors": errors if not is_valid else [],
                    "auto_fixed": True,
                }
                if scenario_meta:
                    result["scenario_metadata"] = scenario_meta
                self._save_to_cache(cache_key, result)
                return result
            except Exception as exc:  # noqa: BLE001
                wait = self._retry_backoff ** attempt
                logger.warning(
                    "Frame %s attempt %d/%d failed: %s (retry in %.1fs)",
                    frame_id, attempt + 1, self._retry_attempts, exc, wait,
                )
                if attempt < self._retry_attempts - 1:
                    time.sleep(wait)

        logger.error("Frame %s: all %d retries exhausted", frame_id, self._retry_attempts)
        return None

    def _call_llm(self, frame: dict, system: str, user: str) -> str:
        """Call LLM (mock or real) and return raw response text.

        Args:
            frame: Frame dict (used to load image for vision API calls).
            system: System prompt.
            user: User prompt.

        Returns:
            Raw text response from the LLM.
        """
        if self._mock is not None:
            return self._mock.annotate(frame, system, user)
        return self._call_anthropic(frame, system, user)

    def _call_anthropic(self, frame: dict, system: str, user: str) -> str:
        """Call Anthropic vision API with image + text message.

        Args:
            frame: Frame dict; ``image_path`` key used to load the image.
            system: System prompt.
            user: User prompt text.

        Returns:
            Response text from Claude.
        """
        content: list[dict] = []
        image_path_str = frame.get("image_path")
        if image_path_str:
            image_path = Path(image_path_str)
            if image_path.exists():
                with image_path.open("rb") as f:
                    image_data = base64.standard_b64encode(f.read()).decode("utf-8")
                media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                })
        content.append({"type": "text", "text": user})

        response = self._client.messages.create(  # type: ignore[union-attr]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Batch annotation
    # ------------------------------------------------------------------

    def annotate_batch(
        self,
        frames: list[dict],
        mode: str = "real",
        max_concurrent: int = 5,
        progress: bool = True,
    ) -> list[dict]:
        """Annotate a batch of frames with async concurrency control.

        Args:
            frames: List of unified frame dicts.
            mode: Annotation mode (``"real"`` or ``"counterfactual"``).
            max_concurrent: Maximum simultaneous in-flight annotations.
            progress: Show tqdm progress bar.

        Returns:
            List of annotation result dicts (failures excluded / logged).
        """
        return asyncio.run(
            self._annotate_batch_async(frames, mode, max_concurrent, progress)
        )

    async def _annotate_batch_async(
        self,
        frames: list[dict],
        mode: str,
        max_concurrent: int,
        progress: bool,
    ) -> list[dict]:
        """Async implementation of batch annotation."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[dict] = []
        checkpoint_every = 50
        bar = tqdm(total=len(frames), desc=f"Annotating ({mode})", disable=not progress)

        async def process(frame: dict) -> dict | None:
            async with semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.annotate_frame, frame, mode)
                bar.update(1)
                return result

        tasks = [process(f) for f in frames]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result is not None:
                results.append(result)
            if (i + 1) % checkpoint_every == 0:
                logger.info("Checkpoint: %d/%d frames annotated", i + 1, len(frames))

        bar.close()
        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        manifest_path: Path,
        output_dir: Path,
        counterfactual_ratio: float | None = None,
    ) -> Path:
        """Run the complete annotation pipeline on a unified manifest.

        Steps:
        1. Load manifest JSONL
        2. Skip already-cached frames
        3. Annotate all frames in "real" mode
        4. Select subset for counterfactual augmentation
        5. Merge annotations into manifest
        6. Save per-split and full annotated manifests
        7. Save quality report and failed-annotation list

        Args:
            manifest_path: Path to the unified manifest JSONL file.
            output_dir: Directory to write all outputs.
            counterfactual_ratio: Override for counterfactual fraction (0–1).

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = _load_manifest(manifest_path)
        logger.info("Loaded %d frames from %s", len(frames), manifest_path)

        # Real annotations
        real_results = self.annotate_batch(frames, mode="real")

        # Counterfactual augmentation
        ratio = counterfactual_ratio if counterfactual_ratio is not None else 0.3
        cf_frames = _select_counterfactual_frames(frames, real_results, ratio)
        cf_results = self.annotate_batch(cf_frames, mode="counterfactual") if cf_frames else []

        # Build annotated frame dicts
        result_map = {r["frame_id"]: r for r in real_results}
        annotated_frames = _merge_annotations(frames, result_map)

        failed = [f for f in frames if f.get("frame_id") not in result_map]

        # Save outputs
        _save_annotated_manifest(annotated_frames, output_dir)
        _save_json(output_dir / "counterfactual_frames.json", cf_results)
        _save_json(output_dir / "failed_annotations.json", [f.get("frame_id") for f in failed])

        report = self.generate_quality_report(real_results + cf_results)
        _save_json(output_dir / "quality_report.json", report)

        logger.info(
            "Pipeline complete: %d annotated, %d counterfactual, %d failed",
            len(real_results), len(cf_results), len(failed),
        )
        return output_dir

    def generate_quality_report(self, results: list[dict]) -> dict:
        """Generate annotation quality metrics across a list of results.

        Args:
            results: List of annotation result dicts from annotate_batch/run_full_pipeline.

        Returns:
            Quality metrics dict.
        """
        total = len(results)
        valid = [r for r in results if not r.get("validation_errors")]
        auto_fixed = [r for r in results if r.get("auto_fixed") and not r.get("validation_errors")]
        real = [r for r in results if r.get("mode") != "counterfactual"]
        cf = [r for r in results if r.get("mode") == "counterfactual"]

        label_dist: dict[str, int] = {}
        severity_dist: dict[str, int] = {}
        hazard_counts: list[int] = []
        reasoning_lens: list[int] = []
        for r in valid:
            ann = r.get("annotation", {})
            hazards = ann.get("hazards", [])
            hazard_counts.append(len(hazards))
            for h in hazards:
                lbl = h.get("label", "unknown")
                label_dist[lbl] = label_dist.get(lbl, 0) + 1
                sev = h.get("severity", "unknown")
                severity_dist[sev] = severity_dist.get(sev, 0) + 1
                reasoning_lens.append(len(h.get("reasoning", "")))

        avg_hazards = (sum(hazard_counts) / len(hazard_counts)) if hazard_counts else 0.0
        avg_reasoning = (sum(reasoning_lens) / len(reasoning_lens)) if reasoning_lens else 0.0

        return {
            "total_frames": total,
            "success_count": len(valid),
            "success_rate": round(len(valid) / total, 4) if total else 0.0,
            "auto_fix_count": len(auto_fixed),
            "auto_fix_rate": round(len(auto_fixed) / total, 4) if total else 0.0,
            "real_annotation_count": len(real),
            "counterfactual_count": len(cf),
            "label_distribution": dict(sorted(label_dist.items())),
            "severity_distribution": dict(sorted(severity_dist.items())),
            "avg_hazards_per_frame": round(avg_hazards, 2),
            "avg_reasoning_length_chars": round(avg_reasoning, 1),
        }


# ---------------------------------------------------------------------------
# SFTDataFormatter
# ---------------------------------------------------------------------------


class SFTDataFormatter:
    """Formats annotated frames into Qwen3-VL chat-format SFT training examples.

    Produces JSONL files where each line is a training example with the
    exact message format that Qwen3-VL's chat template expects.

    Args:
        config: Optional data config dict for prompt customisation.
    """

    SFT_SYSTEM_PROMPT: str = (
        "You are DriveSense, an autonomous vehicle hazard detection system. "
        "Analyze the dashcam image and identify all safety-critical hazards. "
        "Output a structured JSON response with bounding boxes (normalized 0-1000), "
        "hazard classification, severity assessment, reasoning, and recommended action."
    )

    SFT_USER_PROMPT: str = (
        "Analyze this dashcam image for safety hazards. "
        "Identify all hazards with bounding boxes, classify each hazard, "
        "assess severity, explain your reasoning, and recommend an action. "
        "Respond with JSON only."
    )

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def format_single_example(self, frame: dict) -> dict:
        """Format one annotated frame into SFT training format.

        Args:
            frame: Annotated frame dict (must have ``image_path`` and ``annotations``).

        Returns:
            Dict with ``messages``, ``images``, ``frame_id``, and ``source`` keys.
        """
        annotation = frame.get("annotations") or frame.get("annotation") or {}
        return {
            "messages": [
                {"role": "system", "content": self.SFT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame.get("image_path", "")},
                        {"type": "text", "text": self.SFT_USER_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(annotation, ensure_ascii=False),
                },
            ],
            "images": [frame.get("image_path", "")],
            "frame_id": frame.get("frame_id", ""),
            "source": frame.get("source", ""),
        }

    def validate_sft_example(self, example: dict) -> bool:
        """Validate that an SFT example has all required fields.

        Args:
            example: SFT example dict.

        Returns:
            ``True`` if valid, ``False`` otherwise.
        """
        if not isinstance(example.get("messages"), list):
            return False
        messages = example["messages"]
        if len(messages) < 3:
            return False
        roles = [m.get("role") for m in messages]
        if "system" not in roles or "user" not in roles or "assistant" not in roles:
            return False
        return bool(example.get("images"))

    def format_dataset(self, annotated_manifest_path: Path, output_path: Path) -> Path:
        """Format an entire annotated manifest into per-split SFT JSONL files.

        Args:
            annotated_manifest_path: Path to the annotated manifest JSON (full, all splits).
            output_path: Directory to write per-split JSONL files.

        Returns:
            Path to the output directory.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with annotated_manifest_path.open(encoding="utf-8") as fh:
            all_frames = json.load(fh)

        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        skipped = 0
        for frame in all_frames:
            if not (frame.get("annotations") or frame.get("annotation")):
                skipped += 1
                continue
            split = frame.get("split", "train")
            example = self.format_single_example(frame)
            if self.validate_sft_example(example):
                splits.get(split, splits["train"]).append(example)

        stats: dict[str, int] = {}
        for split, examples in splits.items():
            out_file = output_path / f"sft_{split}.jsonl"
            with out_file.open("w", encoding="utf-8") as fh:
                for ex in examples:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats[split] = len(examples)

        stats["skipped_no_annotation"] = skipped
        _save_json(output_path / "sft_format_stats.json", stats)
        logger.info("SFT formatting complete: %s", stats)
        return output_path


# ---------------------------------------------------------------------------
# Internal pipeline helpers
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load frames from a JSONL or JSON manifest file."""
    manifest_path = Path(manifest_path)
    text = manifest_path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _select_counterfactual_frames(
    frames: list[dict],
    real_results: list[dict],
    ratio: float,
) -> list[dict]:
    """Select a subset of successfully annotated frames for counterfactual augmentation.

    Prefers frames whose real annotation contained a genuine hazard (not ``no_hazard``)
    and excludes frames already annotated with ``adverse_weather`` to avoid
    scenario mismatches.

    Args:
        frames: All manifest frames.
        real_results: Successful real annotation results.
        ratio: Fraction of successfully annotated frames to augment.

    Returns:
        List of frame dicts selected for counterfactual annotation.
    """
    success_ids = {r["frame_id"] for r in real_results}
    frame_map = {f.get("frame_id"): f for f in frames}

    # Prefer frames with real hazards (not just no_hazard).
    eligible: list[dict] = []
    for r in real_results:
        if r["frame_id"] not in frame_map:
            continue
        ann = r.get("annotation", {})
        hazards = ann.get("hazards", [])
        labels = {h.get("label") for h in hazards}
        if labels == {"no_hazard"} or labels == {"adverse_weather"}:
            continue
        eligible.append(frame_map[r["frame_id"]])

    if not eligible:
        eligible = [frame_map[fid] for fid in success_ids if fid in frame_map]

    n = max(1, int(len(eligible) * ratio)) if eligible else 0
    random.shuffle(eligible)  # noqa: S311
    return eligible[:n]


def _merge_annotations(frames: list[dict], result_map: dict[str, dict]) -> list[dict]:
    """Merge annotation results back into the frame list."""
    out: list[dict] = []
    for frame in frames:
        fid = frame.get("frame_id")
        result = result_map.get(fid)
        merged = dict(frame)
        if result:
            merged["annotations"] = result.get("annotation")
            merged["annotation_mode"] = result.get("mode", "real")
            merged["annotation_validation_errors"] = result.get("validation_errors", [])
        else:
            merged["annotations"] = None
        out.append(merged)
    return out


def _save_annotated_manifest(annotated_frames: list[dict], output_dir: Path) -> None:
    """Save full + per-split annotated manifests."""
    _save_json(output_dir / "annotated_manifest.json", annotated_frames)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for frame in annotated_frames:
        split = frame.get("split", "train")
        splits.get(split, splits["train"]).append(frame)
    for split, frames in splits.items():
        _save_json(output_dir / f"annotated_manifest_{split}.json", frames)


def _save_json(path: Path, data: object) -> None:
    """Write data as pretty-printed JSON."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
