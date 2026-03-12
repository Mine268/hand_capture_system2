# Step 2 perception notes

## Scope

This step integrates WiLoR-based single-view hand pose estimation into the main C++ project.

## Public API

- `newnewhand::HandPoseEstimator`
  - Input: one BGR `cv::Mat`
  - Output: `std::vector<HandPoseResult>`

## Pipeline

1. `YoloDetector` detects hand boxes and handedness on the full image.
2. Each hand crop is anti-aliased, affine-warped to `256x256`, and flipped when the hand is left.
3. `WilorModel` runs ONNX inference per crop.
4. Output is mirrored back for left hands.
5. `camera_translation` and projected `keypoints_2d` are reconstructed in full-image coordinates.

## Output fields retained for later steps

- detection bbox and confidence
- handedness
- crop center and crop size
- weak-perspective camera `pred_cam`
- full-image `camera_translation`
- `keypoints_2d`
- `keypoints_3d`
- `vertices`
- `global_orient`
- `hand_pose`
- `betas`

These fields are enough for the later stereo calibration and fusion steps.
