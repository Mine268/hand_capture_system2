#include "newnewhand/visualization/hand_pose_overlay.h"

#include <string>

#include <opencv2/imgproc.hpp>

namespace newnewhand {

namespace {

constexpr int kHandConnections[20][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    {0, 17}, {17, 18}, {18, 19}, {19, 20},
};

const cv::Scalar kFingerColors[5] = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
};

}  // namespace

void DrawHandPoseOverlay(
    cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style) {
    for (const auto& result : results) {
        if (style.draw_bbox) {
            cv::rectangle(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.detection.bbox[0]),
                    static_cast<int>(result.detection.bbox[1])),
                cv::Point(
                    static_cast<int>(result.detection.bbox[2]),
                    static_cast<int>(result.detection.bbox[3])),
                cv::Scalar(0, 255, 0),
                style.line_thickness);
        }

        if (style.draw_label) {
            const std::string label = result.detection.is_right ? "R" : "L";
            cv::putText(
                bgr_image,
                label,
                cv::Point(
                    static_cast<int>(result.detection.bbox[0]),
                    static_cast<int>(result.detection.bbox[1]) - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                style.label_scale,
                cv::Scalar(0, 255, 0),
                style.line_thickness);
        }

        for (int connection_index = 0; connection_index < 20; ++connection_index) {
            const int start_joint = kHandConnections[connection_index][0];
            const int end_joint = kHandConnections[connection_index][1];
            cv::line(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[start_joint][0]),
                    static_cast<int>(result.keypoints_2d[start_joint][1])),
                cv::Point(
                    static_cast<int>(result.keypoints_2d[end_joint][0]),
                    static_cast<int>(result.keypoints_2d[end_joint][1])),
                kFingerColors[connection_index / 4],
                style.line_thickness);
        }

        for (int joint_index = 0; joint_index < 21; ++joint_index) {
            cv::circle(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[joint_index][0]),
                    static_cast<int>(result.keypoints_2d[joint_index][1])),
                style.joint_radius,
                cv::Scalar(0, 0, 255),
                -1);
        }
    }
}

cv::Mat RenderHandPoseOverlay(
    const cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style) {
    cv::Mat rendered = bgr_image.clone();
    DrawHandPoseOverlay(rendered, results, style);
    return rendered;
}

}  // namespace newnewhand
