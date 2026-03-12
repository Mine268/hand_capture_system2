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
};

void DrawHandPoseOverlay(
    cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style = {});

cv::Mat RenderHandPoseOverlay(
    const cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style = {});

}  // namespace newnewhand
