# Applications

All examples below assume:

```bash
cd /home/renkaiwen/src/newnewhand
```

Current in-repo inference resources:

- detector: `resources/models/detector.onnx`
- WiLoR backbone: `resources/models/wilor_backbone_opset16.onnx`
- WiLoR backbone external data: `resources/models/wilor_backbone_opset16.onnx.data`
- MANO CPU: `resources/models/mano_cpu_opset16.onnx`
- MANO CPU external data: `resources/models/mano_cpu_opset16.onnx.data`
- MANO mesh faces: `resources/mano_faces.txt`

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

Common options:

- `--cam0_serial <serial>` and `--cam1_serial <serial>` fix stereo camera order
- `--exposure_us <float>` sets exposure
- `--gain <float>` sets manual gain, `-1` means auto gain
- `--fps <int>` throttles capture loop

## Single-view hand pose

### `single_view_hand_pose_demo`

Runs detector + WiLoR + MANO on one image.

Example:

```bash
./build/single_view_hand_pose_demo \
  --image captures/cam0/000001.bmp \
  --gpu
```

Notes:

- `--image <path>` is required
- model paths default to `resources/models/`, so usually no model arguments are needed
- `--gpu` enables CUDA EP for detector and WiLoR backbone
- `--cpu` forces detector and WiLoR backbone onto CPU
- output defaults to `output.png`

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
  --output_dir results/stereo_single_view_pose_split
```

Important options:

- `--gpu` / `--cpu`
- `--preview` / `--no_preview`
- `--save` / `--no_save`
- `--ort_profile <prefix>` writes ONNX Runtime profile JSON for detector/backbone diagnosis
- model paths default to `resources/models/`

## Stereo calibration

### `stereo_calibration_app`

Runs stereo checkerboard calibration from paired images in two directories.

Example:

```bash
./build/stereo_calibration_app \
  --left_dir captures_calib/cam0 \
  --right_dir captures_calib/cam1 \
  --cols 9 \
  --rows 6 \
  --square_size 0.024 \
  --output results/stereo_calibration.yaml \
  --debug_dir results/calib_debug
```

Notes:

- `--left_dir`, `--right_dir`, `--cols`, `--rows`, `--square_size` are required
- left/right images are paired by filename stem
- `--use_sb` is default and preferred
- `--no_use_sb` falls back to the classic detector
- `--preview` opens checkerboard corner preview windows

### `stereo_calibration_visualize_app`

Loads a calibration YAML and renders a third-person scene with `cam0`, `cam1`, and the reconstructed checkerboard poses.

```bash
./build/stereo_calibration_visualize_app \
  --calibration results/stereo_calibration.yaml \
  --output results/stereo_calibration_scene.png \
  --draw_ids
```

## Stereo fusion

### `stereo_fused_hand_pose_demo`

Runs stereo capture, per-view hand pose estimation, and fuses the hand pose into the `cam0` coordinate frame using the stereo calibration.

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --gpu \
  --output_dir results/stereo_fused_hand_pose
```

Notes:

- `--calibration <yaml>` is required
- OpenCV windows still show the per-view overlay images
- a GLFW + OpenGL window renders the fused hand mesh in the `cam0` coordinate frame
- controls in the GLFW window:
  - `W/S`: pitch
  - `A/D`: yaw
  - `Q/E`: roll
  - `I/K`: move scene up/down in view
  - `J/L`: move scene left/right in view
  - `Z/X`: zoom in/out
- use `--no_glfw_view` to disable the GLFW renderer
