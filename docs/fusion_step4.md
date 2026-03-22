# Step 4 fusion notes

## Scope

This step starts stereo fusion in the `cam0` coordinate frame.

## Current strategy

For each handedness:

- use at most one left hand and one right hand per view
- match across views only by handedness
- if both views exist:
  - triangulate the root joint from the two 2D root points
  - keep the `cam0` view WiLoR hand geometry
  - align the `cam0` hand to the triangulated root in `cam0`
- if one view is missing:
  - skip fusion for that handedness

## Output

The demo saves per-frame YAML files containing:

- handedness
- whether the result used stereo fusion
- whether `view0` and `view1` both existed
- root joint in `cam0`
- fused joints in `cam0`
- fused vertices in `cam0`
- MANO parameters

## Offline replay package

The offline fused demo now saves a full frame-by-frame replay package to `--output_dir`.
Use `--offline_dump_dir <dir>` only when you also want a second copy in another location.

Current layout:

- `calibration/stereo_calibration.yaml`
- `manifest.yaml`
- `images/cam0/*.png`
- `images/cam1/*.png`
- `overlays/cam0/*.png`
- `overlays/cam1/*.png`
- `frames/*.yaml`

Each frame YAML contains:

- camera frame metadata for `cam0` and `cam1`
- per-view single-view hand results
- fused hand results in the `cam0` frame
- camera tracking result and trajectory
- mesh vertices
- root joint position
- MANO rotations and shape parameters

## Demo

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --gpu \
  --offline_dump_dir offline_dump/session_001 \
  --output_dir results/stereo_fused_hand_pose
```

## OpenGL viewer

The fused runtime uses a GLFW + OpenGL scene viewer for third-person 3D inspection.

Current viewer content:

- `cam0` coordinate frame
- `cam0` frustum
- calibrated `cam1` coordinate frame
- calibrated `cam1` frustum
- fused hand mesh in the `cam0` frame

Current controls:

- `W/S`: move in world `Z`
- `A/D`: move in world `X`
- `Q/E`: move in world `Y`
- arrow keys: yaw / pitch
- `U/O`: roll
- `Z/X`: zoom
