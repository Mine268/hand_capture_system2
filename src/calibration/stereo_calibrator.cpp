#include "newnewhand/calibration/stereo_calibrator.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace newnewhand {

namespace {

bool HasImageExtension(const std::filesystem::path& path) {
    static const std::set<std::string> kExtensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    };
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return kExtensions.count(ext) > 0;
}

std::string BasenameWithoutExtension(const std::filesystem::path& path) {
    return path.stem().string();
}

cv::Mat EnsureGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace

StereoCalibrator::StereoCalibrator(StereoCalibrationConfig config)
    : config_(std::move(config)) {
    if (config_.checkerboard.inner_corners_cols <= 0 || config_.checkerboard.inner_corners_rows <= 0) {
        throw std::invalid_argument("checkerboard inner corner counts must be positive");
    }
    if (config_.checkerboard.square_size <= 0.0f) {
        throw std::invalid_argument("checkerboard square size must be positive");
    }
}

StereoCalibrationResult StereoCalibrator::Calibrate(const std::vector<CalibrationImagePair>& image_pairs) const {
    if (image_pairs.empty()) {
        throw std::invalid_argument("no calibration image pairs were provided");
    }

    StereoCalibrationResult result;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points_left;
    std::vector<std::vector<cv::Point2f>> image_points_right;

    const std::vector<cv::Point3f> object_corners = BuildObjectCorners();

    for (std::size_t pair_index = 0; pair_index < image_pairs.size(); ++pair_index) {
        const auto& image_pair = image_pairs[pair_index];
        const cv::Mat left_image = cv::imread(image_pair.left_path.string(), cv::IMREAD_COLOR);
        const cv::Mat right_image = cv::imread(image_pair.right_path.string(), cv::IMREAD_COLOR);
        if (left_image.empty()) {
            throw std::runtime_error("failed to read left calibration image: " + image_pair.left_path.string());
        }
        if (right_image.empty()) {
            throw std::runtime_error("failed to read right calibration image: " + image_pair.right_path.string());
        }
        if (left_image.size() != right_image.size()) {
            throw std::runtime_error("left/right calibration image sizes do not match for pair: " + image_pair.left_path.string());
        }

        if (result.image_size.width == 0) {
            result.image_size = left_image.size();
        } else if (left_image.size() != result.image_size) {
            throw std::runtime_error("all calibration images must have the same size");
        }

        StereoCalibrationObservation observation;
        observation.image_pair = image_pair;
        if (!DetectCorners(left_image, right_image, observation.left_corners, observation.right_corners)) {
            continue;
        }

        if (config_.save_debug_images) {
            SaveDebugImage(left_image, right_image, observation, pair_index);
        }

        result.observations.push_back(observation);
        object_points.push_back(object_corners);
        image_points_left.push_back(result.observations.back().left_corners);
        image_points_right.push_back(result.observations.back().right_corners);
    }

    if (result.observations.size() < 3) {
        throw std::runtime_error("at least 3 valid checkerboard observations are required for stereo calibration");
    }

    std::vector<cv::Mat> rvecs_left;
    std::vector<cv::Mat> tvecs_left;
    std::vector<cv::Mat> rvecs_right;
    std::vector<cv::Mat> tvecs_right;

    result.left_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    result.right_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    result.left_dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    result.right_dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

    result.left_rms = cv::calibrateCamera(
        object_points,
        image_points_left,
        result.image_size,
        result.left_camera_matrix,
        result.left_dist_coeffs,
        rvecs_left,
        tvecs_left);
    result.right_rms = cv::calibrateCamera(
        object_points,
        image_points_right,
        result.image_size,
        result.right_camera_matrix,
        result.right_dist_coeffs,
        rvecs_right,
        tvecs_right);

    const int stereo_flags = cv::CALIB_USE_INTRINSIC_GUESS;
    result.stereo_rms = cv::stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        result.left_camera_matrix,
        result.left_dist_coeffs,
        result.right_camera_matrix,
        result.right_dist_coeffs,
        result.image_size,
        result.rotation,
        result.translation,
        result.essential,
        result.fundamental,
        stereo_flags,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6));

    cv::stereoRectify(
        result.left_camera_matrix,
        result.left_dist_coeffs,
        result.right_camera_matrix,
        result.right_dist_coeffs,
        result.image_size,
        result.rotation,
        result.translation,
        result.rectification_left,
        result.rectification_right,
        result.projection_left,
        result.projection_right,
        result.disparity_to_depth);

    result.success = true;
    return result;
}

void StereoCalibrator::SaveResult(const StereoCalibrationResult& result, const std::filesystem::path& output_path) const {
    if (!result.success) {
        throw std::invalid_argument("cannot save an unsuccessful calibration result");
    }

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    cv::FileStorage fs(output_path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open calibration output file: " + output_path.string());
    }

    fs << "image_width" << result.image_size.width;
    fs << "image_height" << result.image_size.height;
    fs << "checkerboard_inner_corners_cols" << config_.checkerboard.inner_corners_cols;
    fs << "checkerboard_inner_corners_rows" << config_.checkerboard.inner_corners_rows;
    fs << "checkerboard_square_size" << config_.checkerboard.square_size;
    fs << "num_valid_pairs" << static_cast<int>(result.observations.size());
    fs << "left_rms" << result.left_rms;
    fs << "right_rms" << result.right_rms;
    fs << "stereo_rms" << result.stereo_rms;
    fs << "left_camera_matrix" << result.left_camera_matrix;
    fs << "right_camera_matrix" << result.right_camera_matrix;
    fs << "left_dist_coeffs" << result.left_dist_coeffs;
    fs << "right_dist_coeffs" << result.right_dist_coeffs;
    fs << "rotation" << result.rotation;
    fs << "translation" << result.translation;
    fs << "essential" << result.essential;
    fs << "fundamental" << result.fundamental;
    fs << "rectification_left" << result.rectification_left;
    fs << "rectification_right" << result.rectification_right;
    fs << "projection_left" << result.projection_left;
    fs << "projection_right" << result.projection_right;
    fs << "disparity_to_depth" << result.disparity_to_depth;

    fs << "valid_pairs" << "[";
    for (const auto& observation : result.observations) {
        fs << "{";
        fs << "left_path" << observation.image_pair.left_path.string();
        fs << "right_path" << observation.image_pair.right_path.string();
        fs << "}";
    }
    fs << "]";
}

std::vector<CalibrationImagePair> StereoCalibrator::CollectImagePairs(
    const std::filesystem::path& left_dir,
    const std::filesystem::path& right_dir) {
    if (!std::filesystem::is_directory(left_dir)) {
        throw std::invalid_argument("left_dir is not a directory: " + left_dir.string());
    }
    if (!std::filesystem::is_directory(right_dir)) {
        throw std::invalid_argument("right_dir is not a directory: " + right_dir.string());
    }

    std::map<std::string, std::filesystem::path> left_images;
    std::map<std::string, std::filesystem::path> right_images;

    for (const auto& entry : std::filesystem::directory_iterator(left_dir)) {
        if (entry.is_regular_file() && HasImageExtension(entry.path())) {
            left_images[BasenameWithoutExtension(entry.path())] = entry.path();
        }
    }
    for (const auto& entry : std::filesystem::directory_iterator(right_dir)) {
        if (entry.is_regular_file() && HasImageExtension(entry.path())) {
            right_images[BasenameWithoutExtension(entry.path())] = entry.path();
        }
    }

    std::vector<CalibrationImagePair> pairs;
    for (const auto& [basename, left_path] : left_images) {
        auto it = right_images.find(basename);
        if (it == right_images.end()) {
            continue;
        }
        pairs.push_back({left_path, it->second});
    }

    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const CalibrationImagePair& lhs, const CalibrationImagePair& rhs) {
            return lhs.left_path.filename().string() < rhs.left_path.filename().string();
        });
    return pairs;
}

bool StereoCalibrator::DetectCorners(
    const cv::Mat& left_image,
    const cv::Mat& right_image,
    std::vector<cv::Point2f>& left_corners,
    std::vector<cv::Point2f>& right_corners) const {
    const cv::Size board_size(
        config_.checkerboard.inner_corners_cols,
        config_.checkerboard.inner_corners_rows);
    const cv::Mat left_gray = EnsureGrayscale(left_image);
    const cv::Mat right_gray = EnsureGrayscale(right_image);

    bool left_found = false;
    bool right_found = false;
    if (config_.use_find_chessboard_sb) {
        const int flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE;
        left_found = cv::findChessboardCornersSB(left_gray, board_size, left_corners, flags);
        right_found = cv::findChessboardCornersSB(right_gray, board_size, right_corners, flags);
    } else {
        const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        left_found = cv::findChessboardCorners(left_gray, board_size, left_corners, flags);
        right_found = cv::findChessboardCorners(right_gray, board_size, right_corners, flags);
        if (left_found) {
            cv::cornerSubPix(
                left_gray,
                left_corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
        }
        if (right_found) {
            cv::cornerSubPix(
                right_gray,
                right_corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
        }
    }

    if (config_.show_detection_preview) {
        cv::Mat left_preview = left_image.clone();
        cv::Mat right_preview = right_image.clone();
        if (left_found) {
            cv::drawChessboardCorners(left_preview, board_size, left_corners, left_found);
        }
        if (right_found) {
            cv::drawChessboardCorners(right_preview, board_size, right_corners, right_found);
        }
        cv::imshow("calibration_left", left_preview);
        cv::imshow("calibration_right", right_preview);
        cv::waitKey(1);
    }

    return left_found && right_found;
}

std::vector<cv::Point3f> StereoCalibrator::BuildObjectCorners() const {
    std::vector<cv::Point3f> object_corners;
    object_corners.reserve(
        static_cast<std::size_t>(config_.checkerboard.inner_corners_cols * config_.checkerboard.inner_corners_rows));

    for (int row = 0; row < config_.checkerboard.inner_corners_rows; ++row) {
        for (int col = 0; col < config_.checkerboard.inner_corners_cols; ++col) {
            object_corners.emplace_back(
                static_cast<float>(col) * config_.checkerboard.square_size,
                static_cast<float>(row) * config_.checkerboard.square_size,
                0.0f);
        }
    }
    return object_corners;
}

void StereoCalibrator::SaveDebugImage(
    const cv::Mat& left_image,
    const cv::Mat& right_image,
    const StereoCalibrationObservation& observation,
    std::size_t pair_index) const {
    if (config_.debug_output_dir.empty()) {
        return;
    }

    std::filesystem::create_directories(config_.debug_output_dir);
    const cv::Size board_size(
        config_.checkerboard.inner_corners_cols,
        config_.checkerboard.inner_corners_rows);

    cv::Mat left_debug = left_image.clone();
    cv::Mat right_debug = right_image.clone();
    cv::drawChessboardCorners(left_debug, board_size, observation.left_corners, true);
    cv::drawChessboardCorners(right_debug, board_size, observation.right_corners, true);

    std::ostringstream left_name;
    left_name << "pair_" << std::setw(4) << std::setfill('0') << pair_index << "_left.png";
    std::ostringstream right_name;
    right_name << "pair_" << std::setw(4) << std::setfill('0') << pair_index << "_right.png";

    cv::imwrite((config_.debug_output_dir / left_name.str()).string(), left_debug);
    cv::imwrite((config_.debug_output_dir / right_name.str()).string(), right_debug);
}

}  // namespace newnewhand
