# demo/examples/

Place sample dashcam images here (`.jpg` or `.png`). Up to 6 images are shown as
clickable example inputs in the Gradio demo gallery.

## How to add images

**Manually**: Copy any dashcam frames you want to showcase into this directory.

**From nuScenes (Colab)**: The `05_demo.ipynb` notebook includes a cell that
automatically copies 6 filtered nuScenes frames here from:

```
outputs/data/nuscenes_filtered/images/
```

## Recommended image properties

- Resolution: 672 × 448 pixels (or 16:9 aspect ratio)
- Content: dashcam frames showing driving scenes
- Format: JPEG or PNG, RGB

## Naming

Files are loaded in sorted order — prefix with `01_`, `02_`, … to control the
display sequence.

Example filenames:
- `01_pedestrian_crossing.jpg`
- `02_vehicle_cut_in.jpg`
- `03_debris_in_lane.jpg`

## Notes

- Images are loaded by `demo/app.py` at startup via `demo/examples/*.{jpg,jpeg,png}`.
- This directory is intentionally kept empty in the repository. Add images locally
  or in your HuggingFace Space before launching.
