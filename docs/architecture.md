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
   - Consumes fused hand pose results and calibration without affecting the core capture or inference pipeline.

## Runtime flow

The intended runtime pipeline is:

1. stereo capture produces a `StereoFrame`
2. WiLoR runs independently on `views[0]` and `views[1]`
3. calibration converts both observations into a common camera frame
4. fusion resolves root pose and stores the final result
5. render optionally visualizes the current frame and pose

## Step 1 status

Step 1 implements the `capture` layer with these design constraints:

- camera `0` and camera `1` are explicitly ordered, preferably by serial number
- the capture library returns owning `cv::Mat` images in BGR format for downstream OpenCV and ONNX code
- the synchronization mechanism is internal to the capture module
- preview and disk saving live only in the demo executable, not in the library
- the public API is RAII-based C++ instead of a global singleton plus C ABI
