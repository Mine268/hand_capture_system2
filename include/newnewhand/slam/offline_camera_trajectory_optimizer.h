#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include "newnewhand/slam/stereo_visual_odometry.h"

namespace newnewhand {

struct OfflineCameraTrajectorySample {
    std::uint64_t capture_index = 0;
    std::chrono::steady_clock::time_point trigger_timestamp;
    StereoCameraTrackingResult initial_tracking_result;
    StereoCameraTrackingResult slam_tracking_result;
    bool has_charuco_pose = false;
    int charuco_num_corners = 0;
    float charuco_reprojection_error_px = 0.0f;
    cv::Matx33f charuco_rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f charuco_camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct OfflineCameraTrajectoryOptimizerConfig {
    double slam_translation_sigma_m = 0.01;
    double slam_rotation_sigma_deg = 1.0;
    double charuco_translation_sigma_m = 0.002;
    double charuco_rotation_sigma_deg = 0.3;
    int min_charuco_corners = 6;
    float max_charuco_reprojection_error_px = 4.0f;
    bool anchor_first_pose_if_unconstrained = true;
    bool verbose_logging = true;
};

struct OfflineCameraTrajectoryOptimizationResult {
    std::vector<StereoCameraTrackingResult> optimized_tracking_results;
    int num_vertices = 0;
    int num_slam_edges = 0;
    int num_charuco_priors = 0;
    bool used_fallback_anchor = false;
};

class OfflineCameraTrajectoryOptimizer {
public:
    explicit OfflineCameraTrajectoryOptimizer(OfflineCameraTrajectoryOptimizerConfig config = {});

    OfflineCameraTrajectoryOptimizationResult Optimize(
        const std::vector<OfflineCameraTrajectorySample>& samples) const;

private:
    OfflineCameraTrajectoryOptimizerConfig config_;
};

}  // namespace newnewhand
