# Annotation pipeline v2

[Back to the project README](../README.md)

Annotation v2 replaced free-form box generation with boxes derived from nuScenes geometry. The
change was made after the first label set collapsed around a few repeated coordinates.

## Why v1 failed

The v1 labeler asked a foundation model to draw boxes directly. It produced:

- 780 full-frame boxes at `[0, 0, 1000, 1000]`;
- 591 copies of `[400, 300, 600, 700]`;
- about 36% of all 4,933 hazards concentrated in only five coordinate tuples.

The validator checked range and ordering but did not reject full-frame boxes, oversized boxes, or
repeated defaults. The trained model reproduced the bad supervision by emitting nearly constant
boxes.

## v2 design

For v2, the foundation model describes hazards but does not localize them. Boxed labels follow this
path:

1. load the nuScenes 3D annotation;
2. transform it into the front-camera coordinate frame;
3. clip cuboid edges against the camera near plane;
4. project the visible geometry into a 2D box;
5. normalize the box to the 0-1000 output coordinate system;
6. validate its area, aspect ratio, dimensions, and frame boundaries;
7. ask Claude for severity, reasoning, and action text for the accepted box.

`high_density` and `no_hazard` are scene-level labels and do not carry boxes.

## Class mapping

| DriveSense label | nuScenes source | Notes |
|---|---|---|
| `occluded_pedestrian` | `human.pedestrian.*` | `visibility_token == 1` |
| `jaywalking` | `human.pedestrian.*` | non-occluded pedestrian; no map-based crossing test |
| `cyclist_proximity` | `vehicle.bicycle`, `vehicle.motorcycle` | projected GT box |
| `construction_zone` | barriers, traffic cones, construction vehicles | projected GT box |
| `unusual_object` | `movable_object.debris` | broader open-world detection is not implemented |
| `high_density` | frame-level density signal | no box; threshold is 15 agents |
| `no_hazard` | no accepted hazard | empty hazard list |

The `jaywalking` label should be read as a pedestrian-road hazard rather than a legal or map-aware
crossing judgment. The current pipeline does not query crosswalk geometry.

## Box filter

The filter rejects a projected box if it:

- covers more than 40% of the image;
- has a side shorter than 1.5% of the image;
- has an aspect ratio outside 0.15-8.0;
- touches three or more frame edges;
- is inverted or entirely outside the camera view.

Every rejection is logged with a reason.

## Dataset validation gate

Run the gate before training:

```bash
python scripts/run_label_validation.py \
  --input-dir outputs/data/sft_ready_v2
```

The command exits non-zero for schema errors or unhealthy box statistics. It checks unique-box
ratio, most-common-box frequency, oversized boxes, box-exempt labels with coordinates, and repeated
coordinates across frames. The repeated-coordinate limit scales with dataset size so that a real
static object seen in adjacent keyframes does not fail a large dataset.

The v1 label set would fail the diversity, frequency, and oversized-box checks.

## Grounding evaluation

Boxed hazards enter IoU matching. `high_density` and `no_hazard` are removed from the box-matching
pool and evaluated by frame-level presence instead. This prevents a scene-level condition from
becoming a false positive or false negative solely because it has no coordinates.

## Regeneration command

`scripts/regenerate_annotations_v2_colab.py` handles curation, GT box sourcing, Batch API
descriptions, SFT output, and validation. It supports a fixed mining shopping list and a per-frame
cache for interrupted API jobs.

Important options:

- `--shopping-list`: label exactly the selected mining records without repeating rarity selection;
- `--max-frames`: small test run before API spending;
- `--dry`: use only image-covered frames and skip paid descriptions;
- `--out-dir`: location for labels, batch state, and resume cache.

The full run rejects an incomplete image mount rather than silently changing the data
distribution. Dense frames use a large response budget because shorter Claude responses were
observed to truncate before valid JSON completed.

## Projection verification

A 50-frame real-data check produced 223 boxes with:

- unique-box ratio: 0.9955;
- most-common-box frequency: 0.009;
- zero boxes above the 40% area limit.

The same sample contained all mapped hazard classes, including 28 occluded pedestrians.

Two bugs were found during this verification:

### Visibility field

nuScenes stores the usable value in `visibility_token` (`"1"` through `"4"`). Parsing the
human-readable `visibility.level` string as an integer caused every pedestrian to fall back to
fully visible. The data loaders and Spark rarity path now use the token.

### Near-plane clipping

The old projector discarded corners behind the camera before projection. A box crossing the camera
plane could therefore become inverted or empty. `project_box_to_2d` now clips every cuboid edge at
`z = 0.1 m` before projection, with unit coverage in `tests/test_transforms.py`.

A separate review of 82 apparently inverted pedestrian boxes showed that all were outside the
front-camera field of view. None was an in-frame pedestrian lost by projection. Near-plane clipping
remains a correctness safeguard rather than a recovery of those 82 cases.

## Compute requirements

GT projection and label validation are CPU-only. Claude descriptions use the API and do not require
a local GPU. Model training is a separate GPU step and begins only after the validation gate passes.
