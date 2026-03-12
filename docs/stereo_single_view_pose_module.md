# Stereo single-view pose module

This module connects the existing stereo capture layer and the single-view WiLoR perception layer.

## Runtime behavior

1. Trigger stereo capture.
2. For camera `0` and camera `1`, run single-view hand pose estimation independently.
3. Keep at most one left hand and one right hand per image, choosing the largest bbox for each handedness.
4. Project the predicted 3D joints back to 2D image coordinates.
5. Draw MANO mesh, 2D skeletons, joints, handedness labels, and bounding boxes on each original image.
6. Render only original-image overlays in the OpenCV preview path.

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

Current default models:

- detector: `resources/models/detector.onnx`
- WiLoR backbone: `resources/models/wilor_backbone_opset16.onnx`
- MANO CPU: `resources/models/mano_cpu_opset16.onnx`

Useful flags:

- `--gpu` / `--cpu`
- `--ort_profile <prefix>`
