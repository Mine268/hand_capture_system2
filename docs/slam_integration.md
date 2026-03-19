# Stereo SLAM Integration Notes

## Goal

Add camera tracking/localization to the current stereo hand-pose runtime without destabilizing the existing capture, calibration, fusion, and viewer flow.

## Mainstream Stereo SLAM Options

### 1. ORB-SLAM3

- Strong baseline for stereo visual SLAM.
- Supports stereo, visual-inertial, and multi-map SLAM.
- Best fit when maximum academic benchmark performance matters.
- Main downside here is integration cost and GPLv3 licensing.

### 2. stella_vslam

- Good practical stereo SLAM option for embedding.
- Supports stereo and map save/load localization workflows.
- Cleaner modular APIs than legacy OpenVSLAM and uses a permissive BSD-style license.
- Best external backend candidate for this repository if long-term reusable maps are required.

### 3. RTAB-Map

- Better choice when graph SLAM, loop closure, multi-session mapping, and ROS ecosystem integration matter more than lightweight embedding.
- Strong for larger robotics stacks, but heavier than this repository currently needs.

### 4. Basalt / OpenVINS / VINS-Fusion

- Best when IMU is available or planned.
- Excellent for visual-inertial odometry and state estimation.
- Less aligned with the current repo because this codebase currently has stereo cameras but no IMU pipeline.

## Current In-Repo Implementation

The repository now includes a first in-tree tracking backend:

- module: `include/newnewhand/slam/` and `src/slam/`
- implementation: stereo visual odometry based on
  - stereo rectification from existing calibration
  - ORB feature extraction
  - left-right descriptor matching and triangulation
  - temporal matching against the previous frame
  - `solvePnPRansac` camera pose estimation

This gives immediate camera tracking/localization with no new external dependency. It is not a full loop-closing SLAM backend yet, but it establishes the pose/trajectory interfaces needed by the runtime and viewer.

## Runtime Integration

`stereo_fused_hand_pose_demo` now:

- tracks camera pose from the same stereo frames used for hand pose
- overlays current SLAM position on the OpenCV preview
- renders SLAM world-origin axes and trajectory in the GLFW viewer

Use:

```bash
./build/stereo_fused_hand_pose_demo --calibration resources/stereo_calibration.yaml --gpu --slam
```

Disable tracking with:

```bash
./build/stereo_fused_hand_pose_demo --calibration resources/stereo_calibration.yaml --gpu --no_slam
```

## Recommended Next Step

If you need true long-term SLAM with loop closure and map reuse, keep the current `slam` module boundary and replace the internal VO backend with `stella_vslam` first. If IMU is added later, evaluate Basalt or OpenVINS.
