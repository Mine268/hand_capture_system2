# Applications

## Capture

### `stereo_capture_demo`

Use for synchronized stereo industrial camera capture.

Example:

```bash
./build/stereo_capture_demo \
  --frames 1 \
  --save \
  --no_preview \
  --output_dir captures
```

## Single-view hand pose

### `single_view_hand_pose_demo`

Runs detector + WiLoR + MANO on one image.

Example:

```bash
./build/single_view_hand_pose_demo \
  --image captures/cam0/000001.bmp \
  --detector_model /home/renkaiwen/src/wilor_deploy/wilor_deploy/WiLoR-mini/wilor_mini/pretrained_models/detector.onnx \
  --wilor_model /home/renkaiwen/src/wilor_deploy/wilor_deploy/onnx_model/wilor_backbone_opset16.onnx \
  --mano_model /home/renkaiwen/src/wilor_deploy/wilor_deploy/onnx_model/mano_cpu_opset16.onnx \
  --gpu
```

## Stereo single-view runtime

### `stereo_single_view_hand_pose_demo`

Runs stereo capture, per-view single-view hand pose estimation, original-image overlays, and third-person mesh preview.

Example:

```bash
./build/stereo_single_view_hand_pose_demo \
  --fps 10 \
  --preview \
  --save \
  --gpu \
  --wilor_model /home/renkaiwen/src/wilor_deploy/wilor_deploy/onnx_model/wilor_backbone_opset16.onnx \
  --mano_model /home/renkaiwen/src/wilor_deploy/wilor_deploy/onnx_model/mano_cpu_opset16.onnx \
  --output_dir results/stereo_single_view_pose_split
```

Shortcuts:

- `q` / `Esc`: quit
- `t`: toggle third-person preview window

## Stereo calibration

### `stereo_calibration_app`

Runs stereo checkerboard calibration from paired images in two directories.

Example:

```bash
./build/stereo_calibration_app \
  --left_dir captures/cam0 \
  --right_dir captures/cam1 \
  --cols 9 \
  --rows 6 \
  --square_size 0.024 \
  --output results/stereo_calibration.yaml
```
