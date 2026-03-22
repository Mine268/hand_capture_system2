#include "newnewhand/slam/stella_vslam_tracker.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "stella_vslam/config.h"
#include "stella_vslam/system.h"

namespace newnewhand {

namespace {

cv::Mat ExtractCameraMatrix(const cv::Mat& projection) {
    return projection.colRange(0, 3).clone();
}

cv::Matx33f ToMatx33f(const stella_vslam::Mat44_t& matrix) {
    cv::Matx33f output = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            output(r, c) = static_cast<float>(matrix(r, c));
        }
    }
    return output;
}

cv::Vec3f ToVec3f(const stella_vslam::Mat44_t& matrix) {
    return cv::Vec3f(
        static_cast<float>(matrix(0, 3)),
        static_cast<float>(matrix(1, 3)),
        static_cast<float>(matrix(2, 3)));
}

std::vector<double> MatToVector(const cv::Mat& matrix) {
    cv::Mat matrix64f;
    matrix.convertTo(matrix64f, CV_64F);
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(matrix64f.rows * matrix64f.cols));
    for (int r = 0; r < matrix64f.rows; ++r) {
        for (int c = 0; c < matrix64f.cols; ++c) {
            values.push_back(matrix64f.at<double>(r, c));
        }
    }
    return values;
}

cv::Mat BuildRectifiedCameraMatrix(const StereoCalibrationResult& calibration) {
    return ExtractCameraMatrix(calibration.projection_left);
}

double BuildFocalXBaseline(const StereoCalibrationResult& calibration) {
    if (calibration.projection_right.empty()) {
        throw std::runtime_error("projection_right is missing from stereo calibration");
    }
    cv::Mat projection64f;
    calibration.projection_right.convertTo(projection64f, CV_64F);
    return std::abs(projection64f.at<double>(0, 3));
}

YAML::Node BuildConfigNode(const StellaStereoSlamTrackerConfig& config) {
    if (!config.calibration.success) {
        throw std::runtime_error("stella tracker requires a valid stereo calibration");
    }

    const cv::Mat rectified_camera_matrix = BuildRectifiedCameraMatrix(config.calibration);
    cv::Mat rectified_camera_matrix64f;
    rectified_camera_matrix.convertTo(rectified_camera_matrix64f, CV_64F);

    YAML::Node root;
    root["System"]["num_grid_cols"] = 48;
    root["System"]["num_grid_rows"] = 36;

    auto camera = root["Camera"];
    camera["name"] = "newnewhand stereo rectified";
    camera["setup"] = "stereo";
    camera["model"] = "perspective";
    camera["fx"] = rectified_camera_matrix64f.at<double>(0, 0);
    camera["fy"] = rectified_camera_matrix64f.at<double>(1, 1);
    camera["cx"] = rectified_camera_matrix64f.at<double>(0, 2);
    camera["cy"] = rectified_camera_matrix64f.at<double>(1, 2);
    camera["k1"] = 0.0;
    camera["k2"] = 0.0;
    camera["p1"] = 0.0;
    camera["p2"] = 0.0;
    camera["k3"] = 0.0;
    camera["fps"] = config.nominal_fps > 0.0 ? config.nominal_fps : 30.0;
    camera["cols"] = config.calibration.image_size.width;
    camera["rows"] = config.calibration.image_size.height;
    camera["focal_x_baseline"] = BuildFocalXBaseline(config.calibration);
    camera["depth_threshold"] = 40.0;
    camera["color_order"] = "BGR";

    root["Preprocessing"]["min_size"] = 800;
    root["Preprocessing"]["num_grid_cols"] = 48;
    root["Preprocessing"]["num_grid_rows"] = 36;

    root["Feature"]["name"] = "newnewhand offline ORB";
    root["Feature"]["scale_factor"] = 1.2;
    root["Feature"]["num_levels"] = 8;
    root["Feature"]["ini_fast_threshold"] = 12;
    root["Feature"]["min_fast_threshold"] = 7;

    root["Tracking"]["margin_last_frame_projection"] = 10.0;
    root["Mapping"]["baseline_dist_thr"] = std::max(0.01, cv::norm(config.calibration.translation));
    root["Mapping"]["redundant_obs_ratio_thr"] = 0.9;
    root["Initializer"]["min_num_triangulated_pts"] = 100;
    return root;
}

void DumpConfigNode(const YAML::Node& config_node, const std::filesystem::path& output_path) {
    if (output_path.empty()) {
        return;
    }
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open stella config output: " + output_path.string());
    }
    ofs << config_node;
}

}  // namespace

struct StellaStereoSlamTracker::Impl {
    explicit Impl(StellaStereoSlamTrackerConfig config_in)
        : config(std::move(config_in)) {
        if (!config.calibration.success) {
            throw std::invalid_argument("stella tracker requires a valid stereo calibration");
        }

        const YAML::Node config_node = BuildConfigNode(config);
        DumpConfigNode(config_node, config.generated_config_dump_path);
        stella_config = std::make_shared<stella_vslam::config>(config_node, config.generated_config_dump_path.string());
        system = std::make_unique<stella_vslam::system>(stella_config, config.vocab_path);
        system->startup(true);
        timestamp_step_seconds = config.nominal_fps > 0.0 ? 1.0 / config.nominal_fps : 1.0 / 30.0;

        rectified_camera_matrix = BuildRectifiedCameraMatrix(config.calibration);
        cv::initUndistortRectifyMap(
            config.calibration.left_camera_matrix,
            config.calibration.left_dist_coeffs,
            config.calibration.rectification_left,
            rectified_camera_matrix,
            config.calibration.image_size,
            CV_32FC1,
            left_map_x,
            left_map_y);
        cv::initUndistortRectifyMap(
            config.calibration.right_camera_matrix,
            config.calibration.right_dist_coeffs,
            config.calibration.rectification_right,
            rectified_camera_matrix,
            config.calibration.image_size,
            CV_32FC1,
            right_map_x,
            right_map_y);
    }

    StereoCameraTrackingResult Track(const StereoFrame& raw_stereo_frame) {
        StereoCameraTrackingResult result;
        result.capture_index = raw_stereo_frame.capture_index;
        result.trigger_timestamp = raw_stereo_frame.trigger_timestamp;
        result.reinitialized = false;

        if (!raw_stereo_frame.is_complete()) {
            result.status_message = "stella stereo frame incomplete";
            return result;
        }

        cv::Mat left_rectified;
        cv::Mat right_rectified;
        cv::remap(
            raw_stereo_frame.views[0].bgr_image,
            left_rectified,
            left_map_x,
            left_map_y,
            cv::INTER_LINEAR);
        cv::remap(
            raw_stereo_frame.views[1].bgr_image,
            right_rectified,
            right_map_x,
            right_map_y,
            cv::INTER_LINEAR);

        const double timestamp_seconds = next_timestamp_seconds;
        next_timestamp_seconds += timestamp_step_seconds;
        const auto pose_wc = system->feed_stereo_frame(left_rectified, right_rectified, timestamp_seconds);

        if (!pose_wc) {
            tracking_lost = true;
            result.status_message = has_pose_once
                ? "stella tracking lost"
                : "stella tracking not initialized";
            return result;
        }

        result.initialized = true;
        result.tracking_ok = true;
        result.reinitialized = has_pose_once && tracking_lost;
        result.rotation_world_from_cam0 = ToMatx33f(*pose_wc);
        result.camera_center_world = ToVec3f(*pose_wc);
        trajectory_world.push_back(result.camera_center_world);
        result.trajectory_world = trajectory_world;
        result.status_message = result.reinitialized
            ? "stella tracking reinitialized"
            : (has_pose_once ? "stella tracking ok" : "stella tracking initialized");

        has_pose_once = true;
        tracking_lost = false;
        return result;
    }

    void Reset() {
        system->shutdown();
        system = std::make_unique<stella_vslam::system>(stella_config, config.vocab_path);
        system->startup(true);
        trajectory_world.clear();
        has_pose_once = false;
        tracking_lost = false;
        next_timestamp_seconds = 0.0;
    }

    StellaStereoSlamTrackerConfig config;
    std::shared_ptr<stella_vslam::config> stella_config;
    std::unique_ptr<stella_vslam::system> system;
    cv::Mat rectified_camera_matrix;
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;
    std::vector<cv::Vec3f> trajectory_world;
    bool has_pose_once = false;
    bool tracking_lost = false;
    double next_timestamp_seconds = 0.0;
    double timestamp_step_seconds = 1.0 / 30.0;
};

StellaStereoSlamTracker::StellaStereoSlamTracker(StellaStereoSlamTrackerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

StellaStereoSlamTracker::~StellaStereoSlamTracker() {
    if (impl_ && impl_->system) {
        impl_->system->shutdown();
    }
}

StereoCameraTrackingResult StellaStereoSlamTracker::Track(const StereoFrame& raw_stereo_frame) {
    return impl_->Track(raw_stereo_frame);
}

void StellaStereoSlamTracker::Reset() {
    impl_->Reset();
}

}  // namespace newnewhand
