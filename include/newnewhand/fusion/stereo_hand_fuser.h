#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

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
    cv::Mat fused_third_person_image;
};

struct StereoHandFuserConfig {
    StereoCalibrationResult calibration;
    bool require_both_views = true;
    bool verbose_logging = true;
};

class StereoHandFuser {
public:
    explicit StereoHandFuser(StereoHandFuserConfig config);

    StereoFusedHandPoseFrame Fuse(const StereoSingleViewPoseFrame& stereo_frame) const;
    void SaveFrame(const StereoFusedHandPoseFrame& frame, const std::filesystem::path& output_path) const;

private:
    FusedHandPose FuseHandByHandedness(const StereoSingleViewPoseFrame& frame, bool is_right) const;
    cv::Vec3f TriangulateRoot(
        const cv::Point2f& point0,
        const cv::Point2f& point1) const;
    void ProjectToView0(HandPoseResult& pose) const;
    void TransformView1PoseToView0(HandPoseResult& pose) const;

    StereoHandFuserConfig config_;
};

}  // namespace newnewhand
