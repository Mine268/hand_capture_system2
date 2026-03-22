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

### `stereo_offline_capture_app`

Captures synchronized stereo frames into a video package for later processing, and can also decode that package back into an image sequence.

Example:

```bash
./build/stereo_offline_capture_app \
  --mode capture \
  --output_dir offline_capture/session_001 \
  --calibration resources/stereo_calibration.yaml \
  --fps 30 \
  --frames 300
```

Notes:

- capture mode writes `videos/cam0.avi`, `videos/cam1.avi`, `manifest.yaml`, and optionally `calibration/stereo_calibration.yaml`
- `--fps` is intended for fixed-rate offline capture such as `30` or `60`
- if `--calibration <yaml>` is provided and contains saved stereo serial numbers, capture order is forced to that calibration
- video writing runs on a background writer thread so capture is less sensitive to disk IO stalls
- `--video_codec MJPG` is the default and is intended to keep capture overhead low
- use decode mode to turn a video package back into `images/cam0` and `images/cam1` for later offline apps:

```bash
./build/stereo_offline_capture_app \
  --mode decode \
  --input_dir offline_capture/session_001 \
  --output_dir offline_capture/session_001_images \
  --image_format png
```

## Stereo tracking

### `stereo_aruco_marker_generator`

Generates a printable ArUco / AprilTag-family marker PNG and an A4 SVG print sheet with exact physical size.

Example:

```bash
./build/stereo_aruco_marker_generator \
  --dictionary DICT_APRILTAG_36h11 \
  --marker_id 0 \
  --size_mm 150
```

Notes:

- default output directory is `resources/markers/`
- the generator writes both a raw PNG and a `_print.svg` sheet for printing
- the first localization demo version defaults to `DICT_APRILTAG_36h11`
- sample generated marker files can be kept under `resources/markers/` for repeatable printing

### `stereo_charuco_board_generator`

Generates a printable ChArUco board PNG and an A4 SVG print sheet with exact physical size.

Example:

```bash
./build/stereo_charuco_board_generator \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 7 \
  --squares_y 5 \
  --square_length_mm 40 \
  --marker_length_mm 30
```

Notes:

- default output directory is `resources/charuco/`
- use the same board parameters later in `stereo_charuco_localization_demo`
- the generator also includes a 100 mm print-check ruler on the SVG page
- sample generated ChArUco boards can be kept under `resources/charuco/` for repeatable experiments

### `stereo_aruco_localization_demo`

Runs live stereo localization from one known ArUco/AprilTag-style marker. The marker coordinate frame is treated as the world frame.

Example:

```bash
./build/stereo_aruco_localization_demo \
  --calibration resources/stereo_calibration.yaml \
  --dictionary DICT_APRILTAG_36h11 \
  --marker_id 0 \
  --marker_length_m 0.12 \
  --preview \
  --glfw_view
```

Notes:

- first version uses OpenCV `aruco` with AprilTag-family dictionaries, no extra third-party detector dependency
- runtime capture is forced to the saved left/right serial numbers in the calibration YAML
- if the marker is seen in the left camera, localization is solved directly in `cam0`
- if only the right camera sees the marker, the pose is converted back into the left-camera frame using the stereo calibration
- no hand pose estimation is involved in this demo
- the GLFW window renders a fixed world frame with the tracked stereo cameras moving inside it

### `stereo_charuco_localization_demo`

Runs live stereo localization from one fixed ChArUco board. The board coordinate frame is treated as the world frame.

Example:

```bash
./build/stereo_charuco_localization_demo \
  --calibration resources/stereo_calibration.yaml \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 7 \
  --squares_y 5 \
  --square_length_m 0.04 \
  --marker_length_m 0.03 \
  --preview \
  --glfw_view
```

Notes:

- this first version localizes from the left camera only
- runtime capture is forced to the saved left/right serial numbers in the calibration YAML
- ChArUco uses many board corners instead of one marker, so it is typically more stable than the single-marker demo
- in practice, reducing `--exposure_us` to around `2000` or `5000` can reduce motion blur and improve tracking stability when the scene is bright enough
- the GLFW window renders a fixed world frame with the stereo cameras moving inside it

### `stereo_camera_tracking_demo`

Runs stereo capture and camera tracking only, using `resources/stereo_calibration.yaml` and the saved left/right camera serial numbers from calibration.

Example:

```bash
./build/stereo_camera_tracking_demo \
  --calibration resources/stereo_calibration.yaml \
  --preview \
  --glfw_view
```

Notes:

- the calibration YAML must contain `left_camera_serial_number` and `right_camera_serial_number`
- runtime capture is forced to those saved serial numbers
- if the connected camera serials do not match the YAML, the app exits before tracking starts
- OpenCV windows show raw left/right images with tracking status
- the GLFW window renders the fixed world frame and the moving tracked stereo cameras

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

Runs stereo capture, per-view single-view hand pose estimation, and original-image overlays.

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

### `stereo_guided_calibration_app`

Interactive guided stereo calibration from live cameras.

Example:

```bash
./build/stereo_guided_calibration_app \
  --output resources/stereo_calibration.yaml
```

Workflow:

- preview the two live views and press `1` if `cam0` is physical left, or `2` if `cam1` is physical left
- enter checkerboard inner-corner cols, rows, and square size in the terminal
- capture 30 valid left-camera checkerboard images at 3 FPS for monocular calibration
- capture 30 valid right-camera checkerboard images at 3 FPS for monocular calibration
- capture 30 valid stereo checkerboard pairs at 3 FPS for stereo extrinsic calibration
- run `calibrateCamera` independently for left and right to estimate intrinsics and distortion
- run stereo calibration with fixed intrinsics/distortion so only the relative extrinsics are optimized
- save calibration parameters plus `left_camera_serial_number` and `right_camera_serial_number` into one YAML

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
- use `stereo_guided_calibration_app` when the physical left/right order is not known in advance

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
  --offline_dump_dir offline_dump/session_001 \
  --output_dir results/stereo_fused_hand_pose
```

Board-localized hand-pose example:

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --gpu \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021
```

Notes:

- `--calibration <yaml>` is required
- if the calibration YAML contains `left_camera_serial_number` and `right_camera_serial_number`, capture order is forced to those serials
- if the active hardware serials do not match the YAML, runtime stops before pose estimation and board localization
- the runtime first applies per-view single-image undistortion from the calibration intrinsics/distortion before ChArUco localization and hand pose estimation; it does not do stereo baseline rectification
- this demo always uses left-view ChArUco board localization; there is no SLAM mode
- ChArUco localization reuses the same calibration YAML and projects the fused hand mesh into the fixed board/world frame in the GLFW viewer
- `--offline_dump_dir <dir>` optionally exports raw stereo views, overlays, calibration, mono results and fused results
- OpenCV windows still show the per-view overlay images
- a GLFW + OpenGL window renders the fused hand mesh in the `cam0` coordinate frame
- the OpenGL scene currently shows:
  - `cam0` at the origin
  - `cam0` axes
  - `cam0` frustum
  - calibrated `cam1` axes and frustum
  - fused hand mesh in the `cam0` frame
- controls in the GLFW window:
  - `W/S`: move in world `Z`
  - `A/D`: move in world `X`
  - `Q/E`: move in world `Y`
  - arrow keys: yaw / pitch
  - `U/O`: roll
  - `Z/X`: zoom in/out
- use `--no_glfw_view` to disable the GLFW renderer

### `stereo_fused_hand_pose_offline_demo`

Loads an offline stereo capture package, runs hand pose estimation and fusion, and uses hybrid localization: ChArUco board pose when available, otherwise SLAM fallback.

```bash
./build/stereo_fused_hand_pose_offline_demo \
  --input_dir offline_capture/session_001 \
  --calibration resources/stereo_calibration.yaml \
  --gpu \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021
```

Notes:

- `--input_dir <dir>` is required and should contain either `images/cam0` + `images/cam1` or `cam0` + `cam1`
- the app first applies per-view single-image undistortion for ChArUco localization and hand pose estimation
- SLAM fallback uses the raw stereo images and performs stereo rectification inside the VO module
- the offline demo uses heavier stereo VO defaults than the live demos: more ORB features, larger stereo search, and more PnP iterations
- if the board is visible, world pose comes from ChArUco; if the board is missing, the app can seed SLAM-to-world alignment from the latest valid calibrated pose and then use aligned SLAM fallback
- `--output_dir <dir>` now saves a full replay package by default: `manifest.yaml`, `calibration/`, `images/`, `overlays/`, and `frames/*.yaml`
- each saved frame yaml contains per-view pose results, fused pose results, and camera tracking data
- `--offline_dump_dir <dir>` optionally writes a second copy of the same replay package to another directory

### `stereo_fused_hand_pose_replay_app`

Replays a saved fused offline package without rerunning pose estimation or camera tracking.

```bash
./build_stella/stereo_fused_hand_pose_replay_app \
  --input_dir results/stereo_fused_hand_pose_offline \
  --fps 30 \
  --preview \
  --glfw_view
```

Notes:

- `--input_dir <dir>` must point to a package written by `stereo_fused_hand_pose_offline_demo`
- 2D preview reuses saved overlays or raw images when overlays are unavailable
- 3D replay uses the saved fused hand poses and saved camera tracking trajectory

Recommended end-to-end workflow:

1. Capture a stereo video package:

```bash
./build/stereo_offline_capture_app \
  --mode capture \
  --output_dir offline_capture/session_001 \
  --calibration resources/stereo_calibration.yaml \
  --fps 30 \
  --frames 300 \
  --video_codec MJPG \
  --gain -1 \
  --no_preview
```

2. Decode the captured video package into an image-sequence package:

```bash
./build/stereo_offline_capture_app \
  --mode decode \
  --input_dir offline_capture/session_001 \
  --output_dir offline_capture/session_001_images \
  --image_format png
```

3. Run offline fused processing on the decoded image sequence:

```bash
./build_stella/stereo_fused_hand_pose_offline_demo \
  --input_dir offline_capture/session_001_images \
  --calibration resources/stereo_calibration.yaml \
  --slam_backend stella \
  --stella_config_dump debug/stella_session_001.yaml \
  --cpu \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021 \
  --verbose \
  --no_preview \
  --no_glfw_view
```
