# DriveSense-VLM — Annotation Pipeline v2 (real boxes)

## Why v2 exists

The v1 auto-annotation loop collapsed the model to constant output because the
**training labels themselves had garbage boxes**. Measured on the generated
label set: 780 full-frame `(0,0,1000,1000)` boxes, a center-blob
`(400,300,600,700)` repeated 591×, and the top-5 boxes accounting for ~36% of
4,933 hazards. The model faithfully learned to emit two constant boxes on every
frame.

### Root cause (v1)
Localization was outsourced to a foundation VLM with no grounding and no size
constraint:

- `prompts/annotation_system.txt` — asks the VLM to "provide Precise bounding box
  coordinates for each hazard" (free-form localization; VLMs default to
  center/full-frame).
- `prompts/annotation_user.txt:19` — **"If NO hazards are present, use label
  `no_hazard` with bbox `[0, 0, 1000, 1000]`"** → manufactured every full-frame box.
- `prompts/counterfactual_user.txt` — "Estimate WHERE this hazard would most
  likely appear" → generic center-blob guesses.
- `annotation.py::_validate_hazard` — validated 4-length / `[0,1000]` / ordering
  only. **No area cap, no catch-all reject**, so a 100%-frame box was "valid".
- `transforms.py::get_2d_bbox_from_3d` (tight nuScenes 3D→2D projection) existed
  but was **never called**.

## v2 principle

**The VLM never localizes.** Boxes come from nuScenes GT geometry (or, later, a
detector for true OOD). The VLM is demoted to *describe-only*: given a real box +
image crop, it produces severity / reasoning / action. Labels come from the box
source, not from the VLM.

## Class → box source (data = nuScenes-only, CAM_FRONT)

| Class | v2 box source | Notes |
|---|---|---|
| occluded_pedestrian | GT projection | `human.pedestrian.*`, `visibility_level == 1` (0–40%) |
| jaywalking | GT projection | `human.pedestrian.*` (non-occluded). **v1 descope:** no map/crosswalk gate — generic pedestrian-hazard box; describe-only VLM may refine. Map lane/ped-crossing logic is a **v2 stretch**. |
| cyclist_proximity | GT projection | `vehicle.bicycle` / `vehicle.motorcycle` |
| construction_zone | GT projection | `movable_object.barrier` / `movable_object.trafficcone` / `vehicle.construction` |
| unusual_object | GT (debris) + detector (OOD) | `movable_object.debris` from GT; true OOD needs OWLv2/GroundingDINO (later) |
| **high_density** | **NONE — scene-level, box-exempt** | Label emitted when the density signal fires (≥15 agents). **No bbox.** Excluded from IoU/box matching; contributes to classification / detection-presence only. |
| no_hazard | NONE | **No box** (v1's full-frame box is deleted) |

## Pipeline

```
nuScenes CAM_FRONT keyframe (rarity-selected)
  │
  ├─[A] BOX SOURCING (no VLM)
  │     per annotation → nuscenes_category_to_hazard(category, visibility)
  │        → get_2d_bbox_from_3d(nusc, ann_token, cam_token)  # tight GT box
  │     high_density → scene-level label, NO box
  │
  ├─[B] HARD BOX FILTER (per frame, box classes only; logs every reject)
  │     reject: area > 40% | degenerate (min side < 1.5%) |
  │             aspect ∉ [0.15, 8] | catch-all (touches ≥3 frame edges)
  │
  ├─[C] VLM DESCRIBE-ONLY (per surviving box): severity/reasoning/action
  │
  └─[D] no_hazard frames → empty hazards list (no box)
```

## Schema change

`high_density` and `no_hazard` are **box-exempt**: a hazard object with one of
these labels carries no `bbox_2d`. All other hazards must carry a filtered GT box.

## Validation gate (runs BEFORE any training)

`scripts/run_label_validation.py` — standalone; **exit 1** on any of:

- `unique_box_ratio` (distinct boxes / total boxes) below threshold (default 0.5)
- `max_single_box_freq` (most common box / total) above **2%**
- any box with area **> 40%** of frame
- `no_hazard`-with-box count **> 0**
- (dataset-level) any single box shared across **> K** distinct frames

Box-exempt labels are excluded from the box statistics. Any one of the first
three would have blocked the v1 dataset.

## Eval change

`grounding.py` treats `BOX_EXEMPT_LABELS = {high_density, no_hazard}` separately:
box-exempt hazards are removed from the IoU matching pool (never FP/FN in box
metrics) and scored by **frame-level presence** (precision/recall) instead.

## Compute

- GT projection: **CPU only** (nuScenes SDK geometry), minutes for all selected
  frames. No GPU.
- OOD detector (unusual_object only, later): OWLv2/GroundingDINO on the
  ~2,754 selected frames ≈ 5–20 min on a 4090, 4–8 GB VRAM.
- VLM describe-only: API, no local GPU.

## Build order (each proven before the next)

1. ✅ This doc.
2. ✅ Wire `get_2d_bbox_from_3d` into box sourcing for GT classes.
3. ✅ Hard box filter + reject log.
4. ✅ Validation gate script (fails the build on the criteria above).
5. ✅ **50-frame proof** on v1.0-trainval (CPU): 0.9955 diversity, all classes
   present, projection verified sound (occluded peds correctly captured; the 82
   dropped are out-of-FOV, not lost). Two bugs found + fixed (visibility_token,
   frustum clipping).
6. ⏭ Full regeneration → gate must pass → retrain. **Approved to plan; not started.**

## Regeneration (`scripts/regenerate_annotations_v2_colab.py`)

One command: curate (`--min-score 5`) → **per-scene dedup** (`--frames-per-scene`,
default 3) → GT box sourcing → describe-only Claude → SFT JSONL (scene-level split)
→ hard gate. Notes from the dry run that shaped it:

- **Describe-only `max_tokens=4096`** — 1024 truncated the JSON on dense (many-
  hazard) frames, causing ~40% parse failures; 4096 fixed it (0 failures).
- **Per-scene dedup** — a static object (barrier/cone) across consecutive keyframes
  of a stopped-ego scene projects to the *same* box on many frames. That's correct
  labelling but temporally-redundant; capping frames per scene removes the near-
  duplicates (and stops the cross-frame-dup gate check from firing on benign repeats).
- **`--dry`** uses image-covered frames only; a non-dry run asserts ≥95% image
  coverage so a partial trainval mount can't silently bias the dataset.
- VLM-failed frames are excluded (never templated); label + bbox_2d always come
  from nuScenes GT.

---

## Step-5 proof results (nuScenes v1.0-trainval, Colab CPU)

Ran the committed `source_boxes_for_frame` / `get_2d_bbox_from_3d` on real frames:

- **Box diversity (representative 50 frames):** `unique_box_ratio = 0.9955`,
  `max_single_box_freq = 0.009`, 223 boxes — vs the v1 labels' 0.33 / 0.16.
  Zero boxes over 40% area. The collapse is gone.
- **Per-class kept (50 frames w/ occluded peds):** jaywalking 97, construction_zone
  75, high_density 44, occluded_pedestrian 28, cyclist_proximity 23, unusual_object 5.
- **Occluded-pedestrian scan (401 vis-1 pedestrians):** 89 kept, 214
  `none_behind_camera` (not in CAM_FRONT — correctly skipped), 82 `inverted_or_zero`,
  16 `degenerate_tiny`.

### Bug found + fixed #1 — visibility parsing
`visibility.level` is a **string** (`"v0-40"`), not an int; `int(level)` fell back
to 4 ("fully visible"), so **`occluded_pedestrian` was never produced** (every
pedestrian became `jaywalking`). Fixed to read `visibility_token` ("1".."4") via
`box_sourcing.visibility_level_of`. Same latent bug still exists in
`nuscenes_loader.py` and `spark_pipeline.py` (rarity occlusion signal) — noted, not
yet fixed.

### Bug found + fixed #2 — near-plane projection (frustum clipping)
The original `get_2d_bbox_from_3d` dropped behind-camera corners and projected only
the front ones. Fixed with proper **near-plane frustum clipping**
(`transforms.project_box_to_2d`): each cuboid edge crossing `z = 0.1 m` is clipped
at the plane before projection. Unit-tested (`tests/test_transforms.py`).

**Re-verification finding (important correction):** the 82 `inverted_or_zero` were
**not** close, straddling pedestrians. A targeted check showed all 82 are
pedestrians in front of the camera but **outside the CAM_FRONT field of view**
(off to the sides/above/below) — they project entirely off-screen and are
*correctly* dropped; **0 were wrongly dropped** (`in_FOV_but_dropped = 0`). So the
frustum clip is a **defensive correctness improvement, not a recovery**: the
pipeline already captured every occluded pedestrian actually visible in CAM_FRONT
(~88 of 400 vis-1 pedestrians; the rest are behind/beside the ego, i.e. not in the
front camera). No close pedestrians were being lost.
