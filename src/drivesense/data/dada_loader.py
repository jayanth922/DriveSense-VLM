"""DADA-2000 dataset loader for critical-moment accident frame extraction.

Implements Phase 1b: scan the DADA-2000 directory structure, load optional
text annotations from an Excel file, extract critical moment and surrounding
context frames for each sequence, resize them, and export a DriveSense-format
manifest with images on disk.

DADA-2000 directory layout expected:
    <dada_root>/
        DADA-2000/
            <category>/          # e.g. "001", "002", ..., "054"
                <sequence>/      # e.g. "001", "002", ...
                    images/
                        <frame>.png   # zero-padded frame filenames

Optional annotation file (Excel):
    <dada_root>/dada_text_annotations.xlsx
    Columns (fuzzy-matched): category, sequence, accident_frame,
    description, weather, time_of_day, road_type

Phase 1b implementation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PILImage
from tqdm import tqdm

if TYPE_CHECKING:
    pass

try:
    import pandas as pd  # type: ignore[import]
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Expected image size for DADA-2000 frames.
_DADA_ORIGINAL_SIZE: tuple[int, int] = (1584, 660)  # width × height


def normalize_column_names(df: object) -> dict[str, str]:
    """Build a mapping from canonical field names to actual DataFrame column names.

    Uses case-insensitive substring matching to handle variations in the Excel
    column headers (e.g. "Accident Frame" → "accident_frame").

    Args:
        df: pandas DataFrame whose columns should be mapped.

    Returns:
        Dict mapping canonical names to matched column names.
        Keys not found in ``df.columns`` are omitted.
    """
    canonical: dict[str, list[str]] = {
        "category":       ["category", "cat"],
        "sequence":       ["sequence", "seq"],
        "accident_frame": ["accident_frame", "accident frame", "frame", "critical"],
        "description":    ["description", "scene", "desc"],
        "weather":        ["weather", "weather_condition", "condition"],
        "time_of_day":    ["time_of_day", "time", "daytime", "lighting"],
        "road_type":      ["road_type", "road", "road type", "location"],
    }
    cols_lower = {c.lower(): c for c in df.columns}  # type: ignore[union-attr]
    mapping: dict[str, str] = {}
    for key, candidates in canonical.items():
        for cand in candidates:
            if cand.lower() in cols_lower:
                mapping[key] = cols_lower[cand.lower()]
                break
    return mapping


def _infer_weather(description: str) -> str:
    """Infer weather condition from a free-text description string.

    Args:
        description: Scene description text.

    Returns:
        One of ``"rain"``, ``"fog"``, ``"snow"``, ``"night"``, or ``"clear"``.
    """
    desc = description.lower()
    for keyword in ("rain", "fog", "snow"):
        if keyword in desc:
            return keyword
    if "night" in desc or "dark" in desc:
        return "night"
    return "clear"


def _infer_time_of_day(description: str) -> str:
    """Infer time of day from a free-text description string.

    Args:
        description: Scene description text.

    Returns:
        One of ``"night"``, ``"dawn"``, ``"dusk"``, or ``"day"``.
    """
    desc = description.lower()
    if "night" in desc or "dark" in desc:
        return "night"
    if "dawn" in desc or "sunrise" in desc:
        return "dawn"
    if "dusk" in desc or "sunset" in desc:
        return "dusk"
    return "day"


def _infer_road_type(description: str) -> str:
    """Infer road type from a free-text description string.

    Args:
        description: Scene description text.

    Returns:
        One of ``"highway"``, ``"intersection"``, ``"parking"``, or ``"urban"``.
    """
    desc = description.lower()
    if "highway" in desc or "motorway" in desc or "freeway" in desc:
        return "highway"
    if "intersection" in desc or "crossing" in desc or "junction" in desc:
        return "intersection"
    if "parking" in desc or "lot" in desc:
        return "parking"
    return "urban"


class DADA2000Loader:
    """Loads and extracts critical-moment frames from the DADA-2000 dataset.

    Scans the directory tree rooted at ``dada_root/DADA-2000/`` to enumerate
    all category/sequence pairs, optionally loads text annotations from an
    Excel file, extracts the critical moment frame plus surrounding context
    frames, resizes them to 672×448, and exports a DriveSense-format manifest.

    Args:
        dada_root: Path to the top-level DADA-2000 download directory
            (must contain a ``DADA-2000/`` sub-directory).
        config: Data config dict loaded from ``configs/data.yaml``.
    """

    def __init__(self, dada_root: Path, config: dict) -> None:
        self.dada_root = Path(dada_root).expanduser()
        self.config = config
        dada_cfg = config.get("dada2000", {})
        fe_cfg = dada_cfg.get("frame_extraction", {})
        self.context_frames: int = int(fe_cfg.get("additional_context_frames", 2))
        self.max_frames: int = int(dada_cfg.get("max_frames", 200))
        preproc = config.get("preprocessing", {})
        target = preproc.get("target_resolution", [672, 448])
        self.target_w: int = int(target[0])
        self.target_h: int = int(target[1])

        self._dataset_dir: Path = self.dada_root / "DADA-2000"
        self._annotations: dict[str, dict] = {}  # key = "cat/seq"
        self._sequences: list[dict] = []

        self._load_annotations()
        self._discover_video_sequences()

    def _load_annotations(self) -> None:
        """Load optional text annotations from ``dada_text_annotations.xlsx``.

        Populates ``self._annotations`` keyed by ``"<category>/<sequence>"``.
        Silently skips if the file is absent or pandas/openpyxl is unavailable.
        """
        xlsx_path = self.dada_root / "dada_text_annotations.xlsx"
        if not xlsx_path.exists():
            logger.debug("Annotation file not found; using inferred metadata: %s", xlsx_path)
            return
        if not _PANDAS_AVAILABLE:
            logger.warning("pandas not installed; skipping annotation file %s", xlsx_path)
            return

        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", xlsx_path, exc)
            return

        col_map = normalize_column_names(df)
        for _, row in df.iterrows():
            cat = str(row[col_map["category"]]).strip().zfill(3) if "category" in col_map else ""
            seq = str(row[col_map["sequence"]]).strip().zfill(3) if "sequence" in col_map else ""
            if not cat or not seq:
                continue
            key = f"{cat}/{seq}"
            entry: dict = {}
            if "accident_frame" in col_map:
                with contextlib.suppress(ValueError, TypeError):
                    entry["accident_frame"] = int(row[col_map["accident_frame"]])
            if "description" in col_map:
                entry["description"] = str(row[col_map["description"]]).strip()
            if "weather" in col_map:
                entry["weather"] = str(row[col_map["weather"]]).strip()
            if "time_of_day" in col_map:
                entry["time_of_day"] = str(row[col_map["time_of_day"]]).strip()
            if "road_type" in col_map:
                entry["road_type"] = str(row[col_map["road_type"]]).strip()
            self._annotations[key] = entry

        logger.info("Loaded annotations for %d sequences", len(self._annotations))

    def _discover_video_sequences(self) -> None:
        """Walk the DADA-2000 directory tree and populate ``self._sequences``.

        Each entry has keys: ``category``, ``sequence``, ``images_dir``,
        ``accident_frame``, ``description``, ``weather``, ``time_of_day``,
        ``road_type``.
        """
        if not self._dataset_dir.exists():
            logger.warning("DADA-2000 dataset dir not found: %s", self._dataset_dir)
            return

        for cat_dir in sorted(self._dataset_dir.iterdir()):
            if not cat_dir.is_dir() or not re.fullmatch(r"\d+", cat_dir.name):
                continue
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                images_dir = seq_dir / "images"
                if not images_dir.exists():
                    continue
                key = f"{cat_dir.name.zfill(3)}/{seq_dir.name.zfill(3)}"
                ann = self._annotations.get(key, {})
                desc = ann.get("description", "")
                entry: dict = {
                    "category": cat_dir.name.zfill(3),
                    "sequence": seq_dir.name.zfill(3),
                    "images_dir": images_dir,
                    "accident_frame": ann.get("accident_frame", None),
                    "description": desc,
                    "weather": ann.get("weather", _infer_weather(desc)),
                    "time_of_day": ann.get("time_of_day", _infer_time_of_day(desc)),
                    "road_type": ann.get("road_type", _infer_road_type(desc)),
                }
                self._sequences.append(entry)

        logger.info("Discovered %d sequences in %s", len(self._sequences), self._dataset_dir)

    def _sorted_frame_paths(self, images_dir: Path) -> list[Path]:
        """Return sorted list of PNG frame paths in an images directory.

        Args:
            images_dir: Path to the ``images/`` sub-directory of a sequence.

        Returns:
            Sorted list of ``.png`` Path objects.
        """
        return sorted(images_dir.glob("*.png"))

    def extract_keyframes(self, seq: dict) -> list[dict]:
        """Extract critical, pre-accident, and mid-accident keyframes for one sequence.

        Frame types extracted:
        - ``critical``: The annotated accident frame (or the last frame if unknown).
        - ``pre_accident``: Up to ``context_frames`` frames before the critical frame.
        - ``mid_accident``: Frames from critical+1 up to ``context_frames`` after.

        Args:
            seq: Sequence dict from ``self._sequences`` (must have ``images_dir``
                and optionally ``accident_frame``).

        Returns:
            List of frame dicts, each with keys:
            ``{"frame_index": int, "frame_type": str, "image": PIL.Image.Image,
               "category": str, "sequence": str, "image_path": Path}``.
            Returns an empty list if no frames are found.
        """
        frame_paths = self._sorted_frame_paths(seq["images_dir"])
        if not frame_paths:
            return []

        n_frames = len(frame_paths)
        acc_idx = seq.get("accident_frame")
        if acc_idx is None:
            # Default to last frame as a reasonable stand-in.
            critical_pos = n_frames - 1
        else:
            # accident_frame is 1-based; clamp to valid range.
            critical_pos = max(0, min(int(acc_idx) - 1, n_frames - 1))

        # Build the set of (position, frame_type) pairs.
        selections: list[tuple[int, str]] = []
        for offset in range(self.context_frames, 0, -1):
            pos = critical_pos - offset
            if pos >= 0:
                selections.append((pos, "pre_accident"))
        selections.append((critical_pos, "critical"))
        for offset in range(1, self.context_frames + 1):
            pos = critical_pos + offset
            if pos < n_frames:
                selections.append((pos, "mid_accident"))

        results: list[dict] = []
        for pos, frame_type in selections:
            path = frame_paths[pos]
            try:
                img = PILImage.open(path).convert("RGB")
                img = self._resize_frame(img)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", path, exc)
                continue
            results.append({
                "frame_index": pos + 1,  # 1-based for consistency
                "frame_type": frame_type,
                "image": img,
                "category": seq["category"],
                "sequence": seq["sequence"],
                "image_path": path,
            })
        return results

    def _resize_frame(self, image: PILImage.Image) -> PILImage.Image:
        """Letterbox-resize a frame to the configured target resolution.

        Args:
            image: Input PIL Image.

        Returns:
            Resized and padded PIL Image of size ``(target_w, target_h)``.
        """
        from drivesense.data.transforms import resize_with_aspect_ratio
        return resize_with_aspect_ratio(image, self.target_w, self.target_h)

    def extract_all_keyframes(self) -> list[dict]:
        """Extract keyframes for all discovered sequences.

        Respects ``self.max_frames`` total frame limit.

        Returns:
            List of frame dicts (see :meth:`extract_keyframes`).
        """
        all_frames: list[dict] = []
        for seq in tqdm(self._sequences, desc="Extracting DADA keyframes", unit="seq"):
            if len(all_frames) >= self.max_frames:
                break
            frames = self.extract_keyframes(seq)
            remaining = self.max_frames - len(all_frames)
            all_frames.extend(frames[:remaining])
        logger.info(
            "Extracted %d keyframes from %d sequences", len(all_frames), len(self._sequences)
        )
        return all_frames

    def export_keyframes(self, output_dir: Path) -> Path:
        """Extract all keyframes, save images, and write a metadata JSONL file.

        Image naming convention:
            ``dada_cat{cat:02d}_seq{seq}_frame{num:04d}_{type}.png``

        Args:
            output_dir: Directory to write images and ``metadata.jsonl``.

        Returns:
            Path to the written ``metadata.jsonl`` file.
        """
        output_dir = Path(output_dir).expanduser()
        images_out = output_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        all_frames = self.extract_all_keyframes()
        jsonl_path = output_dir / "metadata.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for frame in all_frames:
                cat_int = int(frame["category"])
                seq_str = frame["sequence"]
                fname = (
                    f"dada_cat{cat_int:02d}_seq{seq_str}"
                    f"_frame{frame['frame_index']:04d}_{frame['frame_type']}.png"
                )
                dest = images_out / fname
                frame["image"].save(dest, format="PNG")

                # Find the corresponding sequence metadata.
                key = f"{frame['category']}/{frame['sequence']}"
                seq_meta = next(
                    (s for s in self._sequences
                     if f"{s['category']}/{s['sequence']}" == key),
                    {},
                )
                record = {
                    "frame_id": fname[:-4],
                    "source": "dada2000",
                    "image_path": str(dest),
                    "category": frame["category"],
                    "sequence": frame["sequence"],
                    "frame_index": frame["frame_index"],
                    "frame_type": frame["frame_type"],
                    "description": seq_meta.get("description", ""),
                    "weather": seq_meta.get("weather", "clear"),
                    "time_of_day": seq_meta.get("time_of_day", "day"),
                    "road_type": seq_meta.get("road_type", "urban"),
                }
                fh.write(json.dumps(record) + "\n")

        logger.info("Exported %d frames to %s", len(all_frames), output_dir)
        return jsonl_path

    def get_category_distribution(self) -> dict[str, int]:
        """Count the number of sequences per DADA-2000 category.

        Returns:
            Dict mapping category string (zero-padded, e.g. ``"001"``) to
            sequence count.
        """
        dist: dict[str, int] = {}
        for seq in self._sequences:
            dist[seq["category"]] = dist.get(seq["category"], 0) + 1
        return dict(sorted(dist.items()))

    def get_summary_statistics(self) -> dict:
        """Return a summary of the discovered dataset.

        Returns:
            Dict with keys: ``total_sequences``, ``total_categories``,
            ``sequences_with_annotations``, ``max_frames_budget``,
            ``target_resolution``.
        """
        annotated = sum(
            1 for s in self._sequences if s.get("accident_frame") is not None
        )
        return {
            "total_sequences": len(self._sequences),
            "total_categories": len(self.get_category_distribution()),
            "sequences_with_annotations": annotated,
            "max_frames_budget": self.max_frames,
            "target_resolution": [self.target_w, self.target_h],
        }
