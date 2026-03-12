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
- if only `cam0` exists:
  - keep the monocular `cam0` estimate as fallback
- if only `cam1` exists:
  - transform the monocular `cam1` estimate into `cam0`

## Output

The demo saves per-frame YAML files containing:

- handedness
- whether the result used stereo fusion
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
