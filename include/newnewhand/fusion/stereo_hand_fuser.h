#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/perception/hand_pose_estimator.h"
#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"

namespace newnewhand {

struct FusedHandPose {
    HandPoseResult pose_cam0;
    bool fused_from_stereo = false;
    bool is_right = true;
    bool has_view0 = false;
    bool has_view1 = false;
    cv::Vec3f root_joint_cam0 = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct StereoFusedHandPoseFrame {
    std::uint64_t capture_index = 0;
    std::chrono::steady_clock::time_point trigger_timestamp;
    std::vector<FusedHandPose> hands;
};

struct StereoHandFuserConfig {
    StereoCalibrationResult calibration;
    bool require_both_views = true;
    bool verbose_logging = true;
    bool enable_root_kalman = true;
    float root_process_noise = 1e-3f;
    float root_measurement_noise = 4e-4f;
    float root_initial_error = 1e-2f;
    int temporal_reset_frames = 20;
};

class StereoHandFuser {
public:
    explicit StereoHandFuser(StereoHandFuserConfig config);

    StereoFusedHandPoseFrame Fuse(const StereoSingleViewPoseFrame& stereo_frame);
    void SaveFrame(const StereoFusedHandPoseFrame& frame, const std::filesystem::path& output_path) const;

private:
    struct RootFilterState {
        bool initialized = false;
        int missing_frames = 0;
        cv::KalmanFilter filter;
    };

    FusedHandPose FuseHandByHandedness(const StereoSingleViewPoseFrame& frame, bool is_right);
    cv::Vec3f TriangulateRoot(
        const cv::Point2f& point0,
        const cv::Point2f& point1) const;
    void ProjectToView0(HandPoseResult& pose) const;
    void TransformView1PoseToView0(HandPoseResult& pose) const;
    cv::Vec3f FilterRoot(cv::Vec3f root_joint_cam0, bool is_right);
    void MarkMissing(bool is_right);

    StereoHandFuserConfig config_;
    std::array<RootFilterState, 2> root_filters_;
};

}  // namespace newnewhand
