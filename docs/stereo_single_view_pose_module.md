# Stereo single-view pose module

This module connects the existing stereo capture layer and the single-view WiLoR perception layer.

## Runtime behavior

1. Trigger stereo capture.
2. For camera `0` and camera `1`, run single-view hand pose estimation independently.
3. Project the predicted 3D joints back to 2D image coordinates.
4. Draw 2D skeletons, joints, handedness labels, and bounding boxes on each original image.

## Main API

- `newnewhand::StereoSingleViewHandPosePipeline`
  - `Initialize()`
  - `Start()`
  - `CaptureAndEstimate()`
  - `Stop()`
  - `Shutdown()`

`CaptureAndEstimate()` returns `StereoSingleViewPoseFrame`, which contains:

- original per-camera frame metadata
- per-camera `HandPoseResult` list
- per-camera overlay image

## Demo

Use `stereo_single_view_hand_pose_demo` for real capture + per-view pose estimation + overlay preview/save.
