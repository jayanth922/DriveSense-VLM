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

1. This doc.
2. Wire `get_2d_bbox_from_3d` into box sourcing for GT classes.
3. Hard box filter + reject log.
4. Validation gate script (fails the build on the criteria above).
5. **50-frame proof** (box-diversity stats + 5 visual overlays) — requires the
   nuScenes `dataroot`; **no full regeneration / retrain until approved.**
