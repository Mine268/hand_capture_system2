#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"

namespace newnewhand {

struct StereoCameraTrackingResult {
    std::uint64_t capture_index = 0;
    std::chrono::steady_clock::time_point trigger_timestamp;
    bool initialized = false;
    bool tracking_ok = false;
    bool reinitialized = false;
    int left_keypoints = 0;
    int stereo_points = 0;
    int valid_disparity_keypoints = 0;
    int invalid_nonfinite_disparity = 0;
    int invalid_low_disparity = 0;
    int invalid_depth = 0;
    int valid_disparity_pixels = 0;
    float min_valid_disparity = 0.0f;
    float max_valid_disparity = 0.0f;
    int matched_points = 0;
    int tracking_inliers = 0;
    cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
    std::vector<cv::Vec3f> trajectory_world;
    std::string status_message;
};

struct StereoVisualOdometryConfig {
    StereoCalibrationResult calibration;
    bool input_views_are_undistorted = false;
    int max_features = 2000;
    float scale_factor = 1.2f;
    int num_levels = 8;
    int fast_threshold = 12;
    float temporal_ratio_test = 0.85f;
    float min_disparity_px = 1.0f;
    int min_stereo_points = 25;
    int min_tracking_points = 15;
    int stereo_num_disparities = 128;
    int stereo_block_size = 5;
    int stereo_uniqueness_ratio = 8;
    int stereo_speckle_window_size = 50;
    int stereo_speckle_range = 2;
    int stereo_disp12_max_diff = 1;
    int pnp_iterations = 100;
    float pnp_reprojection_error_px = 3.0f;
    double pnp_confidence = 0.99;
    bool verbose_logging = true;
};

class StereoVisualOdometry {
public:
    explicit StereoVisualOdometry(StereoVisualOdometryConfig config);
    ~StereoVisualOdometry();

    StereoVisualOdometry(const StereoVisualOdometry&) = delete;
    StereoVisualOdometry& operator=(const StereoVisualOdometry&) = delete;

    StereoVisualOdometry(StereoVisualOdometry&&) noexcept;
    StereoVisualOdometry& operator=(StereoVisualOdometry&&) noexcept;

    StereoCameraTrackingResult Track(const StereoSingleViewPoseFrame& frame);
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace newnewhand
