"""Unified SFT dataset combining nuScenes rare frames and DADA-2000 accident frames.

Implements Phase 1b: reads frame manifests produced by the nuScenes rarity
pipeline (Parquet or JSONL) and the DADA-2000 extraction pipeline (JSONL),
merges them into a unified frame list, assigns train/val/test splits using
stratified sampling, and exposes a PyTorch-compatible Dataset.

The conversation format follows Qwen3-VL-2B chat template:
    System: "You are a hazard detection assistant for autonomous vehicles..."
    User:   [Image] "Describe any hazards visible in this dashcam frame..."
    Assistant: (structured JSON response)

Phase 1b implementation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

if TYPE_CHECKING:
    pass

# torch.utils.data.Dataset — HPC only; guarded for local macOS dev
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[assignment,misc]

# scikit-learn — optional; used for stratified split
try:
    from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import]
    _SKLEARN_AVAILABLE = True
except ImportError:
    StratifiedShuffleSplit = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

# pandas — optional; used to read Parquet manifests
try:
    import pandas as pd  # type: ignore[import]
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a hazard detection assistant for autonomous vehicles. "
    "Analyse the dashcam frame and identify any hazards present. "
    "Respond only with a JSON object matching the required schema."
)
_USER_PROMPT = (
    "Describe any hazards visible in this dashcam frame. "
    "Return a JSON object with keys: bbox_2d, hazard_class, severity, reasoning, action."
)


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file and return a list of record dicts.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed record dicts; empty if the file does not exist.
    """
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)
    return records


def _load_parquet(path: Path) -> list[dict]:
    """Read a Parquet file (or directory) and return a list of record dicts.

    Args:
        path: Path to a Parquet file or directory of Parquet files.

    Returns:
        List of parsed record dicts; empty if the path does not exist or
        pandas is unavailable.
    """
    if not path.exists() or not _PANDAS_AVAILABLE:
        return []
    try:
        df = pd.read_parquet(str(path))
        return df.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read Parquet %s: %s", path, exc)
        return []


class UnifiedDatasetBuilder:
    """Merge nuScenes and DADA-2000 manifests into a single split manifest.

    Reads frame records from:
    - ``nuscenes_dir/rare_frames/`` (Parquet) or ``nuscenes_dir/metadata.jsonl``
    - ``dada_dir/metadata.jsonl``

    Normalises each record to the unified schema, assigns train/val/test splits
    stratified by ``source+category``, and writes a ``manifest.jsonl`` per split
    under ``output_dir/``.

    Args:
        config: Data config dict loaded from ``configs/data.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        splits_cfg = config.get("splits", {})
        self.train_ratio: float = float(splits_cfg.get("train", 0.8))
        self.val_ratio: float = float(splits_cfg.get("val", 0.1))
        self.seed: int = int(splits_cfg.get("seed", 42))
        self._frames: list[dict] = []

    def load_nuscenes_frames(self, nuscenes_dir: Path) -> int:
        """Load rare nuScenes frames from a Parquet directory or JSONL file.

        Tries ``<nuscenes_dir>/rare_frames/`` (Parquet) first; falls back to
        ``<nuscenes_dir>/metadata.jsonl``.

        Args:
            nuscenes_dir: Base directory of the Spark pipeline output or Phase
                1a filtered output.

        Returns:
            Number of frames loaded.
        """
        nuscenes_dir = Path(nuscenes_dir)
        parquet_dir = nuscenes_dir / "rare_frames"
        jsonl_path = nuscenes_dir / "metadata.jsonl"

        raw: list[dict] = []
        if parquet_dir.exists():
            raw = _load_parquet(parquet_dir)
            logger.info("Loaded %d nuScenes frames from Parquet %s", len(raw), parquet_dir)
        elif jsonl_path.exists():
            raw = _load_jsonl(jsonl_path)
            logger.info("Loaded %d nuScenes frames from JSONL %s", len(raw), jsonl_path)
        else:
            logger.warning("No nuScenes data found in %s", nuscenes_dir)
            return 0

        for rec in raw:
            self._frames.append(self._normalise_nuscenes_record(rec))
        return len(raw)

    def _normalise_nuscenes_record(self, rec: dict) -> dict:
        """Normalise a nuScenes Parquet/JSONL record to the unified schema.

        Args:
            rec: Raw record dict from the nuScenes pipeline.

        Returns:
            Unified frame record dict.
        """
        # nuScenes records from spark pipeline use snake_case keys.
        annotations = rec.get("annotations", []) or []
        cats = list({a.get("category", "") for a in annotations if a.get("category")})
        category = cats[0] if cats else "vehicle"
        return {
            "frame_id": rec.get("sample_token", rec.get("frame_id", "")),
            "source": "nuscenes",
            "image_path": str(rec.get("image_path", "")),
            "scene_description": rec.get("scene_description", rec.get("description", "")),
            "weather": rec.get("weather", "clear"),
            "time_of_day": rec.get("time_of_day", "day"),
            "road_type": rec.get("road_type", "urban"),
            "category": category,
            "rarity_score": int(rec.get("rarity_score", rec.get("total_score", 0))),
            "source_metadata": {
                "scene_token": rec.get("scene_token", ""),
                "sample_token": rec.get("sample_token", ""),
                "camera": rec.get("camera", "CAM_FRONT"),
            },
            "annotations": annotations,
            "split": "",
        }

    def load_dada2000_frames(self, dada_dir: Path) -> int:
        """Load extracted DADA-2000 frames from a metadata JSONL file.

        Args:
            dada_dir: Directory containing ``metadata.jsonl`` written by
                :meth:`DADA2000Loader.export_keyframes`.

        Returns:
            Number of frames loaded.
        """
        jsonl_path = Path(dada_dir) / "metadata.jsonl"
        raw = _load_jsonl(jsonl_path)
        if not raw:
            logger.warning("No DADA-2000 frames found in %s", dada_dir)
            return 0
        logger.info("Loaded %d DADA-2000 frames from %s", len(raw), jsonl_path)
        for rec in raw:
            self._frames.append(self._normalise_dada_record(rec))
        return len(raw)

    def _normalise_dada_record(self, rec: dict) -> dict:
        """Normalise a DADA-2000 JSONL record to the unified schema.

        Args:
            rec: Raw record dict from the DADA-2000 extraction pipeline.

        Returns:
            Unified frame record dict.
        """
        return {
            "frame_id": rec.get("frame_id", ""),
            "source": "dada2000",
            "image_path": str(rec.get("image_path", "")),
            "scene_description": rec.get("description", ""),
            "weather": rec.get("weather", "clear"),
            "time_of_day": rec.get("time_of_day", "day"),
            "road_type": rec.get("road_type", "urban"),
            "category": rec.get("category", "001"),
            "rarity_score": 0,
            "source_metadata": {
                "sequence": rec.get("sequence", ""),
                "frame_index": rec.get("frame_index", 0),
                "frame_type": rec.get("frame_type", "critical"),
            },
            "annotations": [],
            "split": "",
        }

    def assign_splits(self) -> None:
        """Assign train/val/test split labels to all loaded frames.

        Uses :class:`sklearn.model_selection.StratifiedShuffleSplit` with
        stratification on ``source + category``.  Falls back to a sequential
        ratio-based split when scikit-learn is unavailable or the dataset is
        too small.
        """
        n = len(self._frames)
        if n == 0:
            return

        strat_labels = [f"{f['source']}_{f['category']}" for f in self._frames]

        if _SKLEARN_AVAILABLE and n >= 10:
            self._assign_splits_sklearn(strat_labels)
        else:
            self._assign_splits_sequential()

    def _assign_splits_sklearn(self, strat_labels: list[str]) -> None:
        """Assign splits using StratifiedShuffleSplit.

        Args:
            strat_labels: Per-frame stratification label strings.
        """
        import numpy as np

        n = len(self._frames)
        indices = list(range(n))
        # First split: train vs rest
        test_size = round(1.0 - self.train_ratio, 10)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_idx, rest_idx = next(sss1.split(indices, strat_labels))

        # Second split: val vs test from rest
        rest_labels = [strat_labels[i] for i in rest_idx]
        val_frac = self.val_ratio / (1.0 - self.train_ratio)
        val_frac = min(max(val_frac, 0.01), 0.99)

        if len(set(rest_labels)) >= 2 and len(rest_idx) >= 4:
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=1.0 - val_frac, random_state=self.seed
            )
            val_local, test_local = next(sss2.split(list(range(len(rest_idx))), rest_labels))
            val_idx = [rest_idx[i] for i in val_local]
            test_idx = [rest_idx[i] for i in test_local]
        else:
            mid = max(1, int(len(rest_idx) * val_frac))
            val_idx = list(rest_idx[:mid])
            test_idx = list(rest_idx[mid:])

        for i in train_idx:
            self._frames[i]["split"] = "train"
        for i in val_idx:
            self._frames[i]["split"] = "val"
        for i in test_idx:
            self._frames[i]["split"] = "test"

        # Suppress unused import warning
        del np

    def _assign_splits_sequential(self) -> None:
        """Assign splits using simple sequential ratio-based partitioning."""
        n = len(self._frames)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        for i, frame in enumerate(self._frames):
            if i < train_end:
                frame["split"] = "train"
            elif i < val_end:
                frame["split"] = "val"
            else:
                frame["split"] = "test"

    def build(self, output_dir: Path) -> dict[str, Path]:
        """Write per-split ``manifest.jsonl`` files to ``output_dir``.

        Calls :meth:`assign_splits` if splits have not yet been assigned.

        Args:
            output_dir: Directory to write the manifest files.

        Returns:
            Dict mapping split name to manifest ``Path``.
        """
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Assign splits if not done yet.
        if any(f["split"] == "" for f in self._frames):
            self.assign_splits()

        writers: dict[str, Any] = {}
        paths: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            p = output_dir / f"{split}_manifest.jsonl"
            paths[split] = p
            writers[split] = p.open("w", encoding="utf-8")

        try:
            for frame in self._frames:
                split = frame.get("split", "train")
                if split not in writers:
                    split = "train"
                writers[split].write(json.dumps(frame) + "\n")
        finally:
            for fh in writers.values():
                fh.close()

        counts = {s: sum(1 for f in self._frames if f.get("split") == s)
                  for s in ("train", "val", "test")}
        logger.info(
            "Built manifests: train=%d val=%d test=%d → %s",
            counts["train"], counts["val"], counts["test"], output_dir,
        )
        return paths

    def get_statistics(self) -> dict:
        """Return summary statistics of the unified dataset.

        Returns:
            Dict with keys: ``total``, per-split counts, ``sources``,
            ``categories``.
        """
        stats: dict[str, Any] = {
            "total": len(self._frames),
            "train": sum(1 for f in self._frames if f.get("split") == "train"),
            "val":   sum(1 for f in self._frames if f.get("split") == "val"),
            "test":  sum(1 for f in self._frames if f.get("split") == "test"),
            "sources": {},
            "categories": {},
        }
        for f in self._frames:
            stats["sources"][f["source"]] = stats["sources"].get(f["source"], 0) + 1
            stats["categories"][f["category"]] = stats["categories"].get(f["category"], 0) + 1
        return stats

    def print_statistics(self) -> None:
        """Print a human-readable summary of the unified dataset to stdout."""
        s = self.get_statistics()
        print(f"\n[UnifiedDataset] {s['total']} total frames")
        print(f"  train={s['train']}  val={s['val']}  test={s['test']}")
        print(f"  sources: {s['sources']}")
        top_cats = dict(sorted(s["categories"].items(), key=lambda x: -x[1])[:5])
        print(f"  top categories: {top_cats}")


class DriveSenseDataset(TorchDataset):  # type: ignore[misc]
    """SFT dataset for DriveSense-VLM training.

    Loads a per-split ``*_manifest.jsonl`` file produced by
    :class:`UnifiedDatasetBuilder` and returns processed training examples
    ready for the Qwen3-VL processor.

    Args:
        manifest_path: Path to the ``*_manifest.jsonl`` file for this split.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        config: Data config dict loaded from ``configs/data.yaml``.
        processor: Qwen3-VL processor instance (optional; can be set post-init).
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        config: dict,
        processor: object | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.config = config
        self.processor = processor
        self._frames = self.get_split_frames()

    def get_split_frames(self) -> list[dict]:
        """Load and return all frame records from the manifest JSONL.

        Returns:
            List of unified frame dicts for this split.
        """
        return _load_jsonl(self.manifest_path)

    def __len__(self) -> int:
        """Return the number of examples in this split.

        Returns:
            Number of training examples.
        """
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training example.

        Args:
            idx: Integer index into the dataset split.

        Returns:
            Dict with keys:
            - ``"image"``: PIL.Image.Image — dashcam frame (or None if path missing)
            - ``"input_text"``: str — formatted system + user prompt
            - ``"target_text"``: str — placeholder for ground-truth annotation
            - ``"metadata"``: dict — source, split, rarity_score, etc.
        """
        frame = self._frames[idx]
        image = self._load_image(frame.get("image_path", ""))
        input_text = self._format_prompt()
        annotations = frame.get("annotations", []) or []
        target_text = json.dumps(annotations[0]) if annotations else "{}"
        return {
            "image": image,
            "input_text": input_text,
            "target_text": target_text,
            "metadata": {
                "frame_id": frame.get("frame_id", ""),
                "source": frame.get("source", ""),
                "split": frame.get("split", self.split),
                "rarity_score": frame.get("rarity_score", 0),
                "weather": frame.get("weather", ""),
                "time_of_day": frame.get("time_of_day", ""),
                "road_type": frame.get("road_type", ""),
            },
        }

    def _load_image(self, image_path: str) -> PILImage.Image | None:
        """Load an image from disk, returning None on failure.

        Args:
            image_path: Path string to the image file.

        Returns:
            PIL Image in RGB mode, or ``None`` if the file is missing.
        """
        path = Path(image_path)
        if not path.exists():
            return None
        try:
            return PILImage.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load image %s: %s", path, exc)
            return None

    def _format_prompt(self) -> str:
        """Build the system+user prompt string for Qwen3-VL.

        Returns:
            Multi-turn prompt string.
        """
        return f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{_USER_PROMPT}\n<|assistant|>"

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate a list of __getitem__ dicts into a batch dict.

        Images are kept as a list (not tensored) because the VLM processor
        handles dynamic padding.

        Args:
            batch: List of dicts from :meth:`__getitem__`.

        Returns:
            Batched dict with lists for each key.
        """
        return {
            "images":       [item["image"] for item in batch],
            "input_texts":  [item["input_text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
            "metadata":     [item["metadata"] for item in batch],
        }

    def get_collate_fn(self) -> object:
        """Return the static collate function for use with DataLoader.

        Returns:
            :func:`collate_fn` callable.
        """
        return self.collate_fn
