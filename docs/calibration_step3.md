# Step 3 calibration notes

## Scope

This step adds a stereo calibration application based on black-white checkerboard images.

## Checkerboard parameters

The checkerboard definition is passed entirely from the command line:

- `--cols`: number of inner corners along the checkerboard width
- `--rows`: number of inner corners along the checkerboard height
- `--square_size`: physical size of one square edge in your chosen metric unit

## Input convention

The calibration tool expects two directories:

- `left_dir`
- `right_dir`

Images are paired by filename stem, for example:

- `left_dir/000001.png`
- `right_dir/000001.png`

Only matched stems are used.

## Output

The application writes an OpenCV YAML file containing:

- image size
- checkerboard configuration
- number of valid pairs
- left/right intrinsic matrices
- left/right distortion coefficients
- stereo rotation and translation
- essential and fundamental matrices
- stereo rectification matrices

## App

Use:

```bash
./build/stereo_calibration_app \
  --left_dir captures/cam0 \
  --right_dir captures/cam1 \
  --cols 9 \
  --rows 6 \
  --square_size 0.024 \
  --output results/stereo_calibration.yaml
```

Optional:

- `--debug_dir <dir>` saves checkerboard detection overlays
- `--preview` shows corner detection windows
- `--no_use_sb` switches from `findChessboardCornersSB` to the classic corner detector

## Pose visualization

You can also visualize the relative positions of:

- `cam0`
- `cam1`
- each reconstructed checkerboard pose

Use:

```bash
./build/stereo_calibration_visualize_app \
  --calibration results/stereo_calibration.yaml \
  --output results/stereo_calibration_scene.png \
  --draw_ids
```
