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

## Demo

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --gpu \
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
