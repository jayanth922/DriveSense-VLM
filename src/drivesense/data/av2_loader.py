"""Argoverse 2 (AV2) Sensor Dataset loader — mirrors ``nuscenes_loader.py``'s
role for AV2, adapted to av2-api's actual (partially confirmed) data model.

Two clearly separated concerns, matching the confidence split documented in
``docs/AV2_INTEGRATION.md``:

- :class:`AV2LogReader` — the log-IO layer, talking to the real ``av2``
  package's ``AV2SensorDataLoader``. Its exact method names are UNVERIFIED
  this session (research was interrupted before the dataloader's source could
  be fetched) — every call into it is isolated in one small class so a wrong
  method name is a one-place fix once confirmed against the installed package,
  not a rewrite. Stubs raise ``NotImplementedError`` per project convention
  rather than guessing.
- :func:`build_av2_sft_record` — takes ALREADY-LOADED cuboids/camera/pose
  (whatever ``AV2LogReader`` eventually returns, or synthetic test doubles
  with the same confirmed shape) and produces an SFT record in the EXACT
  existing schema, by calling ``SFTDataFormatter.format_single_example``
  directly (not reimplemented) — this is fully testable today without the
  real dataset or even the ``av2`` package installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from drivesense.data.av2_box_sourcing import source_boxes_for_av2_frame

logger = logging.getLogger(__name__)

try:
    from av2.datasets.sensor.av2_sensor_dataloader import (  # type: ignore[import]
        AV2SensorDataLoader,
    )
    _AV2_AVAILABLE = True
except ImportError:
    AV2SensorDataLoader = None  # type: ignore[assignment, misc]
    _AV2_AVAILABLE = False

# The 7 ring cameras (confirmed to exist from the AV2 dataset description;
# exact string spelling below is the documented convention, not independently
# re-verified against source this session).
RING_CAMERAS: tuple[str, ...] = (
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right",
)
FRONT_CAMERA = "ring_front_center"  # closest AV2 analogue to nuScenes' CAM_FRONT


class AV2LogReader:
    """Reads one AV2 log directory: images, cuboids, ego pose, calibration.

    Mirrors ``NuScenesRarityFilter``'s role (a thin, guarded wrapper around the
    vendor SDK) for a single log, since AV2 organizes data per-log-directory
    rather than nuScenes' versioned JSON tables.

    Args:
        dataset_dir: Root directory containing AV2 log directories.
        log_id: The specific log/scene UUID to read.

    Raises:
        ImportError: If the ``av2`` package is not installed.
    """

    def __init__(self, dataset_dir: str | Path, log_id: str) -> None:
        if not _AV2_AVAILABLE:
            raise ImportError("av2 package required. Install with: pip install av2")
        self.dataset_dir = Path(dataset_dir)
        self.log_id = log_id
        # UNVERIFIED: exact AV2SensorDataLoader constructor signature — this
        # session could not fetch its source. Confirm against the installed
        # package; see docs/AV2_INTEGRATION.md.
        self._loader = AV2SensorDataLoader(
            data_dir=self.dataset_dir, labels_dir=self.dataset_dir)

    def get_camera(self, cam_name: str = FRONT_CAMERA) -> object:
        """Return the ``PinholeCamera`` for ``cam_name`` in this log.

        CONFIRMED API: ``PinholeCamera.from_feather(log_dir, cam_name)``.
        """
        from av2.geometry.camera.pinhole_camera import PinholeCamera  # noqa: PLC0415
        return PinholeCamera.from_feather(self.dataset_dir / self.log_id, cam_name)

    def list_timestamps(self, cam_name: str = FRONT_CAMERA) -> list[int]:
        """List camera-synchronized timestamps (nanoseconds) with imagery.

        Raises:
            NotImplementedError: AV2 integration — the exact
                ``AV2SensorDataLoader`` method for this was not confirmed
                against source this session. Confirm and implement before use;
                see docs/AV2_INTEGRATION.md.
        """
        raise NotImplementedError(
            "AV2 integration: AV2LogReader.list_timestamps — exact dataloader "
            "method name unverified this session. See docs/AV2_INTEGRATION.md.")

    def get_cuboids(self, timestamp_ns: int) -> list[object]:
        """Return 3D cuboid annotations at ``timestamp_ns``.

        Raises:
            NotImplementedError: AV2 integration — unverified dataloader
                method; see docs/AV2_INTEGRATION.md.
        """
        raise NotImplementedError(
            "AV2 integration: AV2LogReader.get_cuboids — exact dataloader "
            "method name unverified this session. See docs/AV2_INTEGRATION.md.")

    def get_ego_pose(self, timestamp_ns: int) -> object:
        """Return ``city_SE3_ego`` at ``timestamp_ns``.

        Raises:
            NotImplementedError: AV2 integration — unverified dataloader
                method; see docs/AV2_INTEGRATION.md.
        """
        raise NotImplementedError(
            "AV2 integration: AV2LogReader.get_ego_pose — exact dataloader "
            "method name unverified this session. See docs/AV2_INTEGRATION.md.")

    def get_image_path(self, timestamp_ns: int, cam_name: str = FRONT_CAMERA) -> Path:
        """Return the camera image file path nearest ``timestamp_ns``.

        Raises:
            NotImplementedError: AV2 integration — unverified dataloader
                method; see docs/AV2_INTEGRATION.md.
        """
        raise NotImplementedError(
            "AV2 integration: AV2LogReader.get_image_path — exact dataloader "
            "method name unverified this session. See docs/AV2_INTEGRATION.md.")


# ---------------------------------------------------------------------------
# SFT record building — fully testable today, no av2 package/dataset needed.
# ---------------------------------------------------------------------------


def build_av2_annotation(
    cuboids: list[Any],  # noqa: ANN401 — list[av2.structures.cuboid.Cuboid]
    camera: Any,  # noqa: ANN401 — av2.geometry.camera.pinhole_camera.PinholeCamera
    city_SE3_ego: Any = None,  # noqa: ANN401 — av2.geometry.se3.SE3
    ego_context: dict | None = None,
    density_threshold: int = 15,
) -> dict:
    """Build the ``annotations`` dict for one AV2 frame (GT box/label; no describe pass).

    Severity/reasoning/action are left at placeholder defaults — wiring the
    existing Batch API describe pass (``batch_describe.py``, already built for
    nuScenes) onto AV2 frames is a follow-up, not built this session.

    Args:
        cuboids:            This frame's cuboid list.
        camera:              The target camera.
        city_SE3_ego:        Ego pose, if cuboids are in city frame.
        ego_context:          Optional ``{weather, time_of_day, road_type}``.
        density_threshold:    Passed through to :func:`source_boxes_for_av2_frame`.

    Returns:
        ``{"hazards": [...], "scene_summary": str, "ego_context": {...}}`` —
        the same shape ``SFTDataFormatter.format_single_example`` expects.
    """
    kept, _rejected = source_boxes_for_av2_frame(
        cuboids, camera, city_SE3_ego, density_threshold)
    hazards = [
        {
            "label": h["label"],
            "severity": "medium",  # placeholder pending a describe pass
            "reasoning": "",
            "action": "reduce speed",
            **({"bbox_2d": h["bbox_2d"]} if "bbox_2d" in h else {}),
        }
        for h in kept
    ]
    return {
        "hazards": hazards,
        "scene_summary": "",
        "ego_context": ego_context or {"weather": "unknown", "time_of_day": "unknown",
                                       "road_type": "unknown"},
    }


def build_av2_sft_record(
    image_path: str,
    cuboids: list[Any],  # noqa: ANN401
    camera: Any,  # noqa: ANN401
    frame_id: str,
    city_SE3_ego: Any = None,  # noqa: ANN401
    ego_context: dict | None = None,
) -> dict:
    """Build one AV2 frame into the EXACT existing SFT record schema.

    Reuses ``SFTDataFormatter.format_single_example`` directly (not
    reimplemented), which is what guarantees ``merge_sft_v2.py``,
    ``run_label_validation.py``, and ``run_evaluation.py`` all work unchanged
    on AV2-sourced records — they only ever see this schema, regardless of
    which dataset produced it.

    Args:
        image_path: Path to the camera image file.
        cuboids:    This frame's cuboid list.
        camera:     The target camera.
        frame_id:   A unique frame identifier (e.g. ``f"{log_id}_{timestamp_ns}"``).
        city_SE3_ego: Ego pose, if cuboids are in city frame.
        ego_context:  Optional ``{weather, time_of_day, road_type}``.

    Returns:
        ``{"messages", "images", "frame_id", "source"}`` — identical shape to
        nuScenes-sourced SFT records.
    """
    from drivesense.data.annotation import SFTDataFormatter  # noqa: PLC0415

    annotation = build_av2_annotation(cuboids, camera, city_SE3_ego, ego_context)
    fmt = SFTDataFormatter()
    return fmt.format_single_example({
        "image_path": image_path, "annotations": annotation,
        "frame_id": frame_id, "source": "av2",
    })
