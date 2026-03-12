#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"

namespace newnewhand {

struct StereoCalibrationBoardPose {
    CalibrationImagePair image_pair;
    cv::Matx33f rotation_board_to_cam0 = cv::Matx33f::eye();
    cv::Vec3f translation_board_to_cam0 = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct StereoCalibrationViewStyle {
    int width = 1280;
    int height = 900;
    float yaw_degrees = -35.0f;
    float pitch_degrees = 22.0f;
    float roll_degrees = 0.0f;
    float fit_padding = 1.8f;
    bool draw_axes = true;
    bool draw_board_ids = false;
    cv::Scalar background_color = cv::Scalar(18, 18, 18);
};

class StereoCalibrationVisualizer {
public:
    static std::vector<StereoCalibrationBoardPose> EstimateBoardPoses(
        const StereoCalibrationResult& calibration,
        bool use_find_chessboard_sb = true,
        std::size_t max_pairs = 0);

    static cv::Mat RenderScene(
        const StereoCalibrationResult& calibration,
        const std::vector<StereoCalibrationBoardPose>& board_poses,
        const StereoCalibrationViewStyle& style = {});
};

}  // namespace newnewhand
