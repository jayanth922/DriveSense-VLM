# Argoverse 2 (AV2) integration — design + scaffolding

**Status: scaffolding, not a proven pipeline.** This extends DriveSense-VLM's
hazard-detection pipeline to a second AV dataset (Argoverse 2 Sensor Dataset)
for cross-dataset robustness. The ~1 TB dataset was not downloaded this
session — everything below was built and tested against **synthetic data
matching AV2's documented schema**, the same pattern used for the streaming
miner's synthetic-blob tests. Nothing here has touched a real AV2 image.

## Research: what's confirmed vs. assumed

`pip install av2` was not run in this sandbox, and several follow-up doc
fetches were declined mid-session, so research is **partial**. Every claim
below is tagged with how it was established — treat the ⚠️ tier as "needs
confirming against the installed package before trusting it in a real run."

### ✅ Confirmed — fetched real source from `github.com/argoverse/av2-api`, verbatim, cross-checked

| Component | What's confirmed |
|---|---|
| `av2.geometry.se3.SE3` | `rotation` (3,3), `translation` (3,), `transform_matrix` (4×4 homogeneous), `transform_from(pts)` = `pts @ R.T + t`, `inverse()`, `compose()`. Full file fetched and read verbatim. |
| `av2.structures.cuboid.Cuboid` | `dst_SE3_object: SE3`, `length_m`/`width_m`/`height_m`, `timestamp_ns`, `category`; properties `xyz_center_m`, `dims_lwh_m`, `vertices_m` (cached, 8×3 array of corners "in the destination frame"); `transform()`, `compute_interior_points()`, `from_numpy()`. |
| `av2.geometry.camera.pinhole_camera.PinholeCamera` | `ego_SE3_cam: SE3`, `intrinsics: Intrinsics`, `cam_name: str`; `from_feather(log_dir, cam_name)` classmethod; **`project_ego_to_img(points_ego, remove_nan=False) -> (uv, points_cam, valid_mask)`** — confirmed to do extrinsic transform → `K @ points_cam` → perspective divide → frustum validity mask. `extrinsics` property = `ego_SE3_cam.inverse().transform_matrix`. |
| `Intrinsics` | `fx_px, fy_px, cx_px, cy_px, width_px, height_px`. |
| `CuboidList.project_to_cam()` | Exists, but is **visualization-oriented** (draws colored line segments for a debug overlay) — not a clean bbox extractor. This is why the code below composes `Cuboid.vertices_m` + `PinholeCamera.project_ego_to_img` directly instead. |

### ⚠️ Weak / unverified — from a search-engine-synthesized snippet, not a direct file fetch, or from general pre-training familiarity

| Item | Status |
|---|---|
| `AV2SensorDataLoader` class — exact constructor signature, and method names for listing logs, loading per-timestamp cuboids, loading `city_SE3_ego` | **Not independently confirmed against source this session.** `av2_loader.AV2LogReader` isolates every call to this class in one small file so a wrong method name is a one-place fix, not a rewrite — see the `NotImplementedError` stubs below. |
| `get_log_pinhole_camera(log_id, cam_name)` as a dataloader convenience method | Only from a search snippet — `AV2LogReader.get_camera()` uses the confirmed `PinholeCamera.from_feather()` directly instead, sidestepping the need for this method. |
| Whether a log's loaded cuboid annotations are natively in **ego-vehicle frame or city frame** | The `Cuboid` docstring says "typically ego-vehicle or city frame" without specifying which applies to the standard annotation-loading path. `av2_box_projection.py` takes an explicit, optional `city_SE3_ego` so the caller states which frame applies, rather than the code silently assuming one. **Needs confirming against a real loaded log.** |
| The AV2 30-class category taxonomy (`PEDESTRIAN`, `BICYCLIST`, `REGULAR_VEHICLE`, ...) | From general familiarity with the published Argoverse 2 Sensor Dataset category list, **not re-verified against source this session** (e.g. `av2.utils.metadata` or the official taxonomy docs). `taxonomy_coverage()` is provided specifically so a real category list can be sanity-checked against the table before trusting it. |
| Whether AV2 provides an official small sample/tutorial log for testing without the full download | **Not established this session** — the `av2-api` repo has a `tutorials/` directory (confirmed to exist), but whether it ships a small sample log vs. requiring the full/partial dataset download was not confirmed. Assume you need at least one real log directory; check the tutorials directory yourself before assuming otherwise. |

**Bottom line:** the *geometry* (SE3 composition, pinhole projection, frustum
validity) is built on solid, independently-verified ground. The *log-IO layer*
(how to actually get cuboids/poses/images out of a real AV2 install) and the
*category taxonomy* are best-effort and explicitly flagged — do not run this
against real data without first confirming those two items against the
installed `av2` package.

## What's built

| File | Role | Confidence |
|---|---|---|
| `src/drivesense/data/av2_box_projection.py` | `Cuboid` → 2D bbox, via AV2's own `project_ego_to_img` (not reimplemented) | Built on confirmed API; math verified by tests (perspective scaling, frustum validity, frame correction) |
| `src/drivesense/data/av2_box_sourcing.py` | Category taxonomy mapping + `source_boxes_for_av2_frame` (mirrors `box_sourcing.py`); reuses `box_reject_reason`/`filter_frame_boxes`/`BOX_EXEMPT_LABELS` directly | Taxonomy table is ⚠️ best-effort; box-quality filtering is the same code nuScenes already uses |
| `src/drivesense/data/av2_loader.py` | `AV2LogReader` (log-IO, ⚠️ stubbed pending confirmation) + `build_av2_sft_record` (fully working today) | Split exactly along the confirmed/unverified line |
| `docs/AV2_INTEGRATION.md` | This file | — |

### The key design constraint: identical downstream schema

`build_av2_sft_record()` calls `SFTDataFormatter.format_single_example()`
**directly** — the same function nuScenes' pipeline uses — rather than
reimplementing the SFT record shape. This is what guarantees
`merge_sft_v2.py`, `run_label_validation.py`, and `run_evaluation.py` all work
**unchanged** on AV2 data: they only ever see `{"messages", "images",
"frame_id", "source"}`, regardless of which dataset produced it.

This is proven, not just asserted — `tests/test_av2_loader.py` feeds real
AV2-built records through the actual `run_label_validation.py::collect_stats`
and `merge_sft_v2.py::dedup_by_frame_id`/`assign_scene_split`/
`verify_no_scene_leak` functions and confirms they run cleanly.

**One caller responsibility, not automatic:** `build_av2_sft_record()`
deliberately does **not** stamp `scene_token` or `split` onto the record —
mirroring how nuScenes' own `regenerate_annotations_v2_colab.py` adds those
*after* calling `format_single_example()`, not inside it. Before
`merge_sft_v2.py` can process AV2 records, the (not-yet-built) AV2
annotation-regeneration script must add one line:
```python
rec["scene_token"] = log_id   # AV2's log_id is the natural scene_token analogue
```

### Taxonomy mapping (⚠️ best-effort — see table above)

| AV2 category | Hazard class | Notes |
|---|---|---|
| `PEDESTRIAN`, `WHEELCHAIR`, `STROLLER`, `OFFICIAL_SIGNALER` | `jaywalking` (or `occluded_pedestrian` if an occlusion signal is supplied and low) | |
| `BICYCLIST`, `BICYCLE`, `MOTORCYCLIST`, `MOTORCYCLE`, `WHEELED_RIDER`, `WHEELED_DEVICE` | `cyclist_proximity` | |
| `CONSTRUCTION_CONE`, `CONSTRUCTION_BARREL`, `BOLLARD`, `MESSAGE_BOARD_TRAILER`, `TRAFFIC_LIGHT_TRAILER`, `MOBILE_PEDESTRIAN_CROSSING_SIGN` | `construction_zone` | |
| `DOG`, `ANIMAL`, `STOP_SIGN`, `SIGN` | `unusual_object` | |
| `REGULAR_VEHICLE`, `LARGE_VEHICLE`, `BUS`, `SCHOOL_BUS`, `ARTICULATED_BUS`, `BOX_TRUCK`, `TRUCK`, `TRUCK_CAB`, `VEHICULAR_TRAILER`, `RAILED_VEHICLE` | **deliberately unmapped** (`None`) | Ordinary vehicles aren't hazards in this taxonomy, mirroring `nuscenes_category_to_hazard`'s treatment of `vehicle.car` |
| `high_density`, `no_hazard` | scene-level, not per-category | Same convention as nuScenes: `high_density` fires when cuboid count ≥ threshold; both are box-exempt |
| Any category not in the table at all | **unrecognized**, dropped, logged separately from "deliberately unmapped" | Signals taxonomy drift — check `taxonomy_coverage()` against a real category list before trusting the table |

Occlusion is proxied by an optional `occlusion_signal` int (intended use:
lidar interior-point count via `Cuboid.compute_interior_points()` — a low
count suggests a mostly-occluded object) — this proxy itself is unverified
against real occlusion ground truth.

## Tests

36 tests across `tests/test_av2_box_projection.py`, `tests/test_av2_box_sourcing.py`,
and `tests/test_av2_loader.py`, all synthetic-data-only (`tests/_av2_fakes.py`
implements the *confirmed* `SE3`/`PinholeCamera` math, so the projection tests
exercise real pinhole geometry — verified perspective scaling, frustum
validity, and city↔ego frame correction — not just shape checks). No `av2`
package or real dataset required to run any of them.

## What real data would need to confirm before this is trustworthy

1. **Does `project_ego_to_img` actually produce sane pixel coordinates on a
   real image?** The synthetic tests prove the *orchestration* is internally
   consistent (frame conversion, frustum handling, AABB extraction all behave
   correctly relative to each other); they cannot prove AV2's real intrinsics/
   extrinsics/coordinate conventions match what's assumed here. **This
   requires a real log.**
2. **Ego vs. city frame** — confirm which frame `Cuboid.vertices_m` uses for a
   normally-loaded log's annotations, and whether `city_SE3_ego` needs to be
   supplied in practice.
3. **`AV2LogReader`'s stubbed methods** (`list_timestamps`, `get_cuboids`,
   `get_ego_pose`, `get_image_path`) — implement against the real
   `AV2SensorDataLoader` API once confirmed; currently raise
   `NotImplementedError` rather than guessing.
4. **The category taxonomy** — run `taxonomy_coverage()` against a real log's
   category list; a nonzero "unrecognized" count means the table needs updating.
5. **Wiring the existing Batch API describe pass** (`batch_describe.py`,
   already built for nuScenes) onto AV2 frames — not attempted this session;
   `build_av2_annotation()` currently leaves severity/reasoning/action at
   placeholder defaults.

None of this is a small technicality — until (1) and (2) are confirmed against
a real log, this is a well-tested piece of *plumbing*, not a validated
detection pipeline. Treat it accordingly.
