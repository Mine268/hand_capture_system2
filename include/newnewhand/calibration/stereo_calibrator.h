#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace newnewhand {

struct CheckerboardConfig {
    int inner_corners_cols = 0;
    int inner_corners_rows = 0;
    float square_size = 0.0f;
};

struct CalibrationImagePair {
    std::filesystem::path left_path;
    std::filesystem::path right_path;
};

struct StereoCalibrationConfig {
    CheckerboardConfig checkerboard;
    bool use_find_chessboard_sb = true;
    bool show_detection_preview = false;
    bool save_debug_images = false;
    std::filesystem::path debug_output_dir;
    std::size_t max_image_pairs = 0;
};

struct StereoCalibrationObservation {
    CalibrationImagePair image_pair;
    std::vector<cv::Point2f> left_corners;
    std::vector<cv::Point2f> right_corners;
};

struct StereoCalibrationResult {
    bool success = false;
    cv::Size image_size;
    CheckerboardConfig checkerboard;
    std::vector<StereoCalibrationObservation> observations;

    double left_rms = 0.0;
    double right_rms = 0.0;
    double stereo_rms = 0.0;

    cv::Mat left_camera_matrix;
    cv::Mat right_camera_matrix;
    cv::Mat left_dist_coeffs;
    cv::Mat right_dist_coeffs;
    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat essential;
    cv::Mat fundamental;
    cv::Mat rectification_left;
    cv::Mat rectification_right;
    cv::Mat projection_left;
    cv::Mat projection_right;
    cv::Mat disparity_to_depth;
};

class StereoCalibrator {
public:
    explicit StereoCalibrator(StereoCalibrationConfig config);

    StereoCalibrationResult Calibrate(const std::vector<CalibrationImagePair>& image_pairs) const;
    void SaveResult(const StereoCalibrationResult& result, const std::filesystem::path& output_path) const;
    static StereoCalibrationResult LoadResult(const std::filesystem::path& input_path);

    static std::vector<CalibrationImagePair> CollectImagePairs(
        const std::filesystem::path& left_dir,
        const std::filesystem::path& right_dir);

private:
    bool DetectCorners(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        std::vector<cv::Point2f>& left_corners,
        std::vector<cv::Point2f>& right_corners) const;

    std::vector<cv::Point3f> BuildObjectCorners() const;
    void SaveDebugImage(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        const StereoCalibrationObservation& observation,
        std::size_t pair_index) const;

    StereoCalibrationConfig config_;
};

}  // namespace newnewhand
