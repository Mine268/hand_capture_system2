#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace newnewhand {

struct HandDetection {
    float bbox[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float confidence = 0.0f;
    bool is_right = true;
};

struct HandPoseResult {
    HandDetection detection;
    float crop_center[2] = {0.0f, 0.0f};
    float crop_size = 0.0f;
    float focal_length_px = 0.0f;
    float pred_cam[3] = {0.0f, 0.0f, 0.0f};
    float camera_translation[3] = {0.0f, 0.0f, 0.0f};
    float keypoints_2d[21][2] = {};
    float keypoints_3d[21][3] = {};
    float vertices[778][3] = {};
    float global_orient[3] = {0.0f, 0.0f, 0.0f};
    float hand_pose[15][3] = {};
    float betas[10] = {};
};

struct HandPoseEstimatorConfig {
    std::string detector_model_path;
    std::string wilor_model_path;
    std::string debug_dump_dir;
    bool use_gpu = true;
    float detection_confidence_threshold = 0.3f;
    float detection_nms_threshold = 0.45f;
    float focal_length = 5000.0f;
    int patch_size = 256;
    float crop_rescale_factor = 2.5f;
};

class HandPoseEstimator {
public:
    explicit HandPoseEstimator(HandPoseEstimatorConfig config);
    ~HandPoseEstimator();

    HandPoseEstimator(const HandPoseEstimator&) = delete;
    HandPoseEstimator& operator=(const HandPoseEstimator&) = delete;

    HandPoseEstimator(HandPoseEstimator&&) noexcept;
    HandPoseEstimator& operator=(HandPoseEstimator&&) noexcept;

    std::vector<HandPoseResult> Predict(const cv::Mat& bgr_image);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace newnewhand
