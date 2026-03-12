#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

#include "newnewhand/perception/hand_pose_estimator.h"

namespace newnewhand {

struct HandPoseOverlayStyle {
    int line_thickness = 2;
    int joint_radius = 3;
    double label_scale = 0.7;
    bool draw_bbox = true;
    bool draw_label = true;
    bool draw_mesh = true;
    bool draw_mesh_wireframe = false;
    double mesh_alpha = 0.72;
};

struct ThirdPersonHandViewStyle {
    int width = 640;
    int height = 640;
    float yaw_degrees = -35.0f;
    float pitch_degrees = 18.0f;
    float roll_degrees = 0.0f;
    float fit_padding = 1.8f;
    bool draw_mesh = true;
    bool draw_wireframe = true;
    cv::Scalar background_color = cv::Scalar(18, 18, 18);
};

void DrawHandPoseOverlay(
    cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style = {});

cv::Mat RenderHandPoseOverlay(
    const cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style = {});

cv::Mat RenderThirdPersonHandMeshView(
    const std::vector<HandPoseResult>& results,
    const ThirdPersonHandViewStyle& style = {});

}  // namespace newnewhand
