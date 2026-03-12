# newnewhand architecture

## Overall modules

The project is split into five layers so each step can be developed and verified independently.

1. `capture`
   - Responsible only for stereo industrial camera enumeration, configuration, synchronized triggering, and frame delivery.
   - Exposes stable C++ data objects such as `CameraDescriptor`, `CameraFrame`, and `StereoFrame`.
   - Does not know anything about WiLoR, calibration, or rendering.

2. `perception`
   - Runs single-view hand detection and WiLoR inference on a single image.
   - Produces per-view hand results with joints, mesh, MANO parameters, and confidence metadata.

3. `calibration`
   - Loads and validates intrinsic and extrinsic calibration.
   - Provides projection, triangulation, and coordinate-frame conversion utilities.

4. `fusion`
   - Combines stereo observations with calibration and root-joint constraints.
   - Produces a final hand pose referenced to camera `0`.
   - Owns result serialization.

5. `render`
   - Optional real-time visualization for image overlays and mesh rendering.
   - Split into:
     - OpenCV overlay rendering on top of captured images
     - GLFW + OpenGL scene rendering for interactive third-person inspection
   - Consumes fused hand pose results and calibration without affecting the core capture or inference pipeline.

## Runtime flow

The intended runtime pipeline is:

1. stereo capture produces a `StereoFrame`
2. WiLoR runs independently on `views[0]` and `views[1]`
3. calibration converts both observations into a common camera frame
4. fusion resolves root pose and stores the final result
5. render optionally visualizes the current frame and pose

## Current status

Implemented:

- step 1 capture layer
- step 2 single-view WiLoR perception
- stereo single-view runtime integration
- 2D overlay and MANO mesh overlay on captured images
- step 3 stereo checkerboard calibration application
- stereo calibration pose visualization
- step 4 initial stereo fusion in the `cam0` frame
- GLFW + OpenGL interactive scene viewer for fused 3D inspection

Current application entry points:

- `stereo_capture_demo`
- `single_view_hand_pose_demo`
- `stereo_single_view_hand_pose_demo`
- `stereo_calibration_app`
- `stereo_calibration_visualize_app`
- `stereo_fused_hand_pose_demo`
