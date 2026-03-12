#include "newnewhand/perception/hand_pose_estimator.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hand_pose_utils.h"
#include "wilor_model.h"
#include "yolo_detector.h"

namespace newnewhand {

struct HandPoseEstimator::Impl {
    explicit Impl(HandPoseEstimatorConfig config_in)
        : config(std::move(config_in)),
          detector(config.detector_model_path),
          wilor(config.wilor_model_path, config.use_gpu) {
        if (config.patch_size != 256) {
            throw std::invalid_argument("current WiLoR ONNX pipeline requires patch_size == 256");
        }
    }

    HandPoseEstimatorConfig config;
    YoloDetector detector;
    WilorModel wilor;
};

HandPoseEstimator::HandPoseEstimator(HandPoseEstimatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

HandPoseEstimator::~HandPoseEstimator() = default;

HandPoseEstimator::HandPoseEstimator(HandPoseEstimator&&) noexcept = default;

HandPoseEstimator& HandPoseEstimator::operator=(HandPoseEstimator&&) noexcept = default;

std::vector<HandPoseResult> HandPoseEstimator::Predict(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        throw std::invalid_argument("input image must not be empty");
    }

    const auto raw_detections = impl_->detector.Detect(
        bgr_image,
        impl_->config.detection_confidence_threshold,
        impl_->config.detection_nms_threshold);

    const float image_width = static_cast<float>(bgr_image.cols);
    const float image_height = static_cast<float>(bgr_image.rows);
    const float image_extent = std::max(image_width, image_height);

    std::vector<Detection> detections;
    detections.reserve(raw_detections.size());
    for (const auto& detection : raw_detections) {
        Detection clamped = detection;
        clamped.x1 = std::clamp(clamped.x1, 0.0f, image_width - 1.0f);
        clamped.y1 = std::clamp(clamped.y1, 0.0f, image_height - 1.0f);
        clamped.x2 = std::clamp(clamped.x2, 0.0f, image_width - 1.0f);
        clamped.y2 = std::clamp(clamped.y2, 0.0f, image_height - 1.0f);

        const float bbox_width = clamped.x2 - clamped.x1;
        const float bbox_height = clamped.y2 - clamped.y1;
        if (bbox_width < 8.0f || bbox_height < 8.0f) {
            continue;
        }

        detections.push_back(clamped);
    }

    if (detections.empty()) {
        return {};
    }

    struct HandCropInfo {
        HandDetection detection;
        float center_x = 0.0f;
        float center_y = 0.0f;
        float bbox_size = 0.0f;
    };

    std::vector<HandCropInfo> hand_crops;
    std::vector<cv::Mat> patches;
    hand_crops.reserve(detections.size());
    patches.reserve(detections.size());

    for (const auto& detection : detections) {
        HandCropInfo hand_crop;
        hand_crop.detection.bbox[0] = detection.x1;
        hand_crop.detection.bbox[1] = detection.y1;
        hand_crop.detection.bbox[2] = detection.x2;
        hand_crop.detection.bbox[3] = detection.y2;
        hand_crop.detection.confidence = detection.confidence;
        hand_crop.detection.is_right = detection.class_id == 1;
        hand_crop.center_x = (detection.x1 + detection.x2) * 0.5f;
        hand_crop.center_y = (detection.y1 + detection.y2) * 0.5f;

        const float bbox_width = detection.x2 - detection.x1;
        const float bbox_height = detection.y2 - detection.y1;
        hand_crop.bbox_size = std::max(
            impl_->config.crop_rescale_factor * bbox_width,
            impl_->config.crop_rescale_factor * bbox_height);

        const cv::Mat blurred = hand_pose_utils::AntiAliasBlur(
            bgr_image,
            hand_crop.bbox_size,
            impl_->config.patch_size);
        const cv::Mat patch = hand_pose_utils::GenerateImagePatch(
            blurred,
            hand_crop.center_x,
            hand_crop.center_y,
            hand_crop.bbox_size,
            impl_->config.patch_size,
            !hand_crop.detection.is_right);

        hand_crops.push_back(hand_crop);
        patches.push_back(patch);
    }

    const WilorOutput wilor_output = impl_->wilor.Infer(patches);
    const float scaled_focal_length =
        impl_->config.focal_length / static_cast<float>(impl_->config.patch_size) * image_extent;

    std::vector<HandPoseResult> results;
    results.reserve(hand_crops.size());

    for (int hand_index = 0; hand_index < static_cast<int>(hand_crops.size()); ++hand_index) {
        const HandCropInfo& hand_crop = hand_crops[hand_index];

        HandPoseResult result;
        result.detection = hand_crop.detection;
        result.crop_center[0] = hand_crop.center_x;
        result.crop_center[1] = hand_crop.center_y;
        result.crop_size = hand_crop.bbox_size;

        float pred_cam[3] = {
            wilor_output.pred_cam[hand_index * 3 + 0],
            wilor_output.pred_cam[hand_index * 3 + 1],
            wilor_output.pred_cam[hand_index * 3 + 2],
        };

        if (!hand_crop.detection.is_right) {
            pred_cam[1] = -pred_cam[1];
        }

        std::memcpy(
            result.vertices,
            wilor_output.pred_vertices.data() + hand_index * 778 * 3,
            sizeof(float) * 778 * 3);
        std::memcpy(
            result.keypoints_3d,
            wilor_output.pred_keypoints_3d.data() + hand_index * 21 * 3,
            sizeof(float) * 21 * 3);
        std::memcpy(
            result.global_orient,
            wilor_output.global_orient.data() + hand_index * 3,
            sizeof(float) * 3);
        std::memcpy(
            result.hand_pose,
            wilor_output.hand_pose.data() + hand_index * 15 * 3,
            sizeof(float) * 15 * 3);
        std::memcpy(
            result.betas,
            wilor_output.betas.data() + hand_index * 10,
            sizeof(float) * 10);
        std::memcpy(result.pred_cam, pred_cam, sizeof(pred_cam));

        if (!hand_crop.detection.is_right) {
            for (int joint_index = 0; joint_index < 21; ++joint_index) {
                result.keypoints_3d[joint_index][0] = -result.keypoints_3d[joint_index][0];
            }
            for (int vertex_index = 0; vertex_index < 778; ++vertex_index) {
                result.vertices[vertex_index][0] = -result.vertices[vertex_index][0];
            }
            result.global_orient[1] = -result.global_orient[1];
            result.global_orient[2] = -result.global_orient[2];
            for (int pose_index = 0; pose_index < 15; ++pose_index) {
                result.hand_pose[pose_index][1] = -result.hand_pose[pose_index][1];
                result.hand_pose[pose_index][2] = -result.hand_pose[pose_index][2];
            }
        }

        const std::array<float, 3> camera_translation = hand_pose_utils::CamCropToFull(
            pred_cam,
            hand_crop.center_x,
            hand_crop.center_y,
            hand_crop.bbox_size,
            image_width,
            image_height,
            scaled_focal_length);
        std::memcpy(result.camera_translation, camera_translation.data(), sizeof(float) * 3);

        float flat_keypoints_2d[21 * 2] = {};
        hand_pose_utils::PerspectiveProjection(
            &result.keypoints_3d[0][0],
            21,
            result.camera_translation,
            scaled_focal_length,
            image_width * 0.5f,
            image_height * 0.5f,
            flat_keypoints_2d);
        std::memcpy(result.keypoints_2d, flat_keypoints_2d, sizeof(float) * 21 * 2);

        results.push_back(result);
    }

    return results;
}

}  // namespace newnewhand
