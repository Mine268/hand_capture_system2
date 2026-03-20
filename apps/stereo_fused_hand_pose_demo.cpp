#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include "newnewhand/fusion/stereo_hand_fuser.h"
#include "newnewhand/io/offline_sequence_writer.h"
#include "newnewhand/render/glfw_scene_viewer.h"
#include "newnewhand/slam/stereo_visual_odometry.h"

namespace {

std::string ProjectRoot() {
#ifdef NEWNEWHAND_PROJECT_ROOT
    return NEWNEWHAND_PROJECT_ROOT;
#else
    return ".";
#endif
}

std::string DefaultDetectorModelPath() { return ProjectRoot() + "/resources/models/detector.onnx"; }
std::string DefaultWilorModelPath() { return ProjectRoot() + "/resources/models/wilor_backbone_opset16.onnx"; }
std::string DefaultManoModelPath() { return ProjectRoot() + "/resources/models/mano_cpu_opset16.onnx"; }
std::string DefaultCalibrationPath() { return ProjectRoot() + "/resources/stereo_calibration.yaml"; }

std::string FormatPoint(float x, float y) {
    std::ostringstream oss;
    oss << "(" << x << ", " << y << ")";
    return oss.str();
}

struct DemoOptions {
    std::string output_dir = "results/stereo_fused_hand_pose";
    std::string offline_dump_dir;
    std::string debug_dir = "debug/wilor_failures";
    std::string ort_profile_prefix;
    std::string calibration_path = DefaultCalibrationPath();
    std::string cam0_serial;
    std::string cam1_serial;
    std::string detector_model_path = DefaultDetectorModelPath();
    std::string wilor_model_path = DefaultWilorModelPath();
    std::string mano_model_path = DefaultManoModelPath();
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 10;
    int frames = -1;
    bool preview = true;
    bool save = true;
    bool use_gpu = true;
    bool verbose = true;
    bool glfw_view = true;
    bool slam = false;
    bool charuco_localization = false;
    std::string dictionary_name = "DICT_APRILTAG_36h11";
    int squares_x = 5;
    int squares_y = 7;
    float square_length_m = 0.028f;
    float marker_length_m = 0.021f;
    bool legacy_pattern = false;
};

struct CharucoDetectionResult {
    bool detected = false;
    int num_charuco_corners = 0;
    float reprojection_error_px = 0.0f;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat charuco_corners;
    cv::Mat charuco_ids;
    cv::Mat marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
};

struct LatestFrameData {
    newnewhand::StereoSingleViewPoseFrame stereo_frame;
    newnewhand::StereoFusedHandPoseFrame fused_frame;
    newnewhand::StereoCameraTrackingResult tracking_result;
    CharucoDetectionResult charuco_detection;
};

struct SharedWorkerState {
    std::mutex mutex;
    std::shared_ptr<const LatestFrameData> latest_frame;
    std::exception_ptr worker_error;
};

struct CharucoLocalizationState {
    bool has_pose = false;
    cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
    std::vector<cv::Vec3f> trajectory_world;
};

DemoOptions ParseArgs(int argc, char** argv) {
    DemoOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };

        if (arg == "--output_dir") options.output_dir = require_value(arg);
        else if (arg == "--debug_dir") options.debug_dir = require_value(arg);
        else if (arg == "--offline_dump_dir") options.offline_dump_dir = require_value(arg);
        else if (arg == "--ort_profile") options.ort_profile_prefix = require_value(arg);
        else if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--cam0_serial") options.cam0_serial = require_value(arg);
        else if (arg == "--cam1_serial") options.cam1_serial = require_value(arg);
        else if (arg == "--detector_model") options.detector_model_path = require_value(arg);
        else if (arg == "--wilor_model") options.wilor_model_path = require_value(arg);
        else if (arg == "--mano_model") options.mano_model_path = require_value(arg);
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--fps") options.fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--save") options.save = true;
        else if (arg == "--no_save") options.save = false;
        else if (arg == "--glfw_view") options.glfw_view = true;
        else if (arg == "--no_glfw_view") options.glfw_view = false;
        else if (arg == "--slam") options.slam = true;
        else if (arg == "--no_slam") options.slam = false;
        else if (arg == "--charuco_localization") options.charuco_localization = true;
        else if (arg == "--no_charuco_localization") options.charuco_localization = false;
        else if (arg == "--dictionary") options.dictionary_name = require_value(arg);
        else if (arg == "--squares_x") options.squares_x = std::stoi(require_value(arg));
        else if (arg == "--squares_y") options.squares_y = std::stoi(require_value(arg));
        else if (arg == "--square_length_m") options.square_length_m = std::stof(require_value(arg));
        else if (arg == "--marker_length_m") options.marker_length_m = std::stof(require_value(arg));
        else if (arg == "--legacy_pattern") options.legacy_pattern = true;
        else if (arg == "--no_legacy_pattern") options.legacy_pattern = false;
        else if (arg == "--gpu") options.use_gpu = true;
        else if (arg == "--cpu") options.use_gpu = false;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--quiet") options.verbose = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_fused_hand_pose_demo [options]\n"
                << "  --calibration <path>      default: " << DefaultCalibrationPath() << "\n"
                << "  --detector_model <path>   default: " << DefaultDetectorModelPath() << "\n"
                << "  --wilor_model <path>      default: " << DefaultWilorModelPath() << "\n"
                << "  --mano_model <path>       default: " << DefaultManoModelPath() << "\n"
                << "  --output_dir <dir>        default: results/stereo_fused_hand_pose\n"
                << "  --offline_dump_dir <dir>  default: disabled\n"
                << "  --save | --no_save        default: --save\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
                << "  --slam | --no_slam        default: --no_slam\n"
                << "  --charuco_localization | --no_charuco_localization  default: --no_charuco_localization\n"
                << "  --dictionary <name>       default: DICT_APRILTAG_36h11\n"
                << "  --squares_x <int>         default: 5\n"
                << "  --squares_y <int>         default: 7\n"
                << "  --square_length_m <f>     default: 0.028\n"
                << "  --marker_length_m <f>     default: 0.021\n"
                << "  --verbose | --quiet       default: --verbose\n"
                << "  --gpu | --cpu             default: --gpu\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void SaveImage(
    const std::filesystem::path& root,
    const std::string& subdir,
    std::uint64_t capture_index,
    const cv::Mat& image) {
    std::filesystem::create_directories(root / subdir);
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << ".png";
    cv::imwrite((root / subdir / name.str()).string(), image);
}

void SaveFusedYaml(
    const newnewhand::StereoHandFuser& fuser,
    const newnewhand::StereoFusedHandPoseFrame& frame,
    const std::filesystem::path& root) {
    std::filesystem::create_directories(root / "yaml");
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << frame.capture_index << ".yaml";
    fuser.SaveFrame(frame, root / "yaml" / name.str());
}

void ApplyCalibrationSerialsToCaptureConfig(
    const DemoOptions& options,
    const newnewhand::StereoCalibrationResult& calibration,
    newnewhand::StereoSingleViewHandPosePipelineConfig& pipeline_config) {
    const bool has_saved_serials =
        !calibration.left_camera_serial_number.empty() && !calibration.right_camera_serial_number.empty();
    if (!has_saved_serials) {
        pipeline_config.capture_config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        return;
    }

    if (!options.cam0_serial.empty() && options.cam0_serial != calibration.left_camera_serial_number) {
        throw std::runtime_error(
            "requested --cam0_serial does not match calibration left_camera_serial_number");
    }
    if (!options.cam1_serial.empty() && options.cam1_serial != calibration.right_camera_serial_number) {
        throw std::runtime_error(
            "requested --cam1_serial does not match calibration right_camera_serial_number");
    }

    pipeline_config.capture_config.serial_numbers = {
        calibration.left_camera_serial_number,
        calibration.right_camera_serial_number,
    };
}

void ValidateActiveCamerasAgainstCalibration(
    const std::array<newnewhand::CameraDescriptor, 2>& active_cameras,
    const newnewhand::StereoCalibrationResult& calibration) {
    const bool has_saved_serials =
        !calibration.left_camera_serial_number.empty() && !calibration.right_camera_serial_number.empty();
    if (!has_saved_serials) {
        return;
    }

    if (active_cameras[0].serial_number != calibration.left_camera_serial_number
        || active_cameras[1].serial_number != calibration.right_camera_serial_number) {
        std::ostringstream oss;
        oss
            << "active stereo serials do not match calibration yaml: expected left="
            << calibration.left_camera_serial_number
            << " right=" << calibration.right_camera_serial_number
            << " but got cam0=" << active_cameras[0].serial_number
            << " cam1=" << active_cameras[1].serial_number;
        throw std::runtime_error(oss.str());
    }
}

int ParseDictionaryName(const std::string& name) {
    static const std::unordered_map<std::string, int> kDictionaryMap = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
        {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
        {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    const auto it = kDictionaryMap.find(name);
    if (it == kDictionaryMap.end()) {
        throw std::runtime_error("unsupported dictionary: " + name);
    }
    return it->second;
}

cv::Vec3f ToVec3f(const cv::Mat& vector) {
    cv::Mat vector32f;
    vector.convertTo(vector32f, CV_32F);
    return cv::Vec3f(
        vector32f.at<float>(0, 0),
        vector32f.at<float>(1, 0),
        vector32f.at<float>(2, 0));
}

cv::Vec3f MatVecMul(const cv::Matx33f& matrix, const cv::Vec3f& v) {
    return cv::Vec3f(
        matrix(0, 0) * v[0] + matrix(0, 1) * v[1] + matrix(0, 2) * v[2],
        matrix(1, 0) * v[0] + matrix(1, 1) * v[1] + matrix(1, 2) * v[2],
        matrix(2, 0) * v[0] + matrix(2, 1) * v[1] + matrix(2, 2) * v[2]);
}

cv::Mat BuildUndistortedCameraMatrix(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Size& image_size) {
    return cv::getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        0.0,
        image_size);
}

float MeanCornerReprojectionError(
    const cv::Mat& object_points,
    const cv::Mat& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    std::vector<cv::Point3f> object_points_vec;
    std::vector<cv::Point2f> image_points_vec;
    object_points.copyTo(object_points_vec);
    image_points.copyTo(image_points_vec);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points_vec, rvec, tvec, camera_matrix, dist_coeffs, projected);
    if (projected.size() != image_points_vec.size() || projected.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    float error = 0.0f;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        error += cv::norm(projected[i] - image_points_vec[i]);
    }
    return error / static_cast<float>(projected.size());
}

CharucoDetectionResult EstimateCharucoPoseInLeftCamera(
    const cv::Mat& undistorted_left_image,
    const cv::aruco::CharucoBoard& board,
    const cv::aruco::CharucoDetector& detector,
    const cv::Mat& undistorted_camera_matrix,
    const cv::Mat& undistorted_zero_dist_coeffs) {
    CharucoDetectionResult result;
    detector.detectBoard(
        undistorted_left_image,
        result.charuco_corners,
        result.charuco_ids,
        result.marker_corners,
        result.marker_ids);

    if (result.charuco_ids.empty()) {
        return result;
    }

    result.num_charuco_corners = result.charuco_ids.total();
    if (result.num_charuco_corners < 4 || board.checkCharucoCornersCollinear(result.charuco_ids)) {
        return result;
    }

    cv::Mat object_points;
    cv::Mat image_points;
    board.matchImagePoints(result.charuco_corners, result.charuco_ids, object_points, image_points);
    if (object_points.empty() || image_points.empty()) {
        return result;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    const bool pnp_ok = cv::solvePnP(
        object_points,
        image_points,
        undistorted_camera_matrix,
        undistorted_zero_dist_coeffs,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_ITERATIVE);
    if (!pnp_ok) {
        return result;
    }

    cv::solvePnPRefineLM(
        object_points,
        image_points,
        undistorted_camera_matrix,
        undistorted_zero_dist_coeffs,
        rvec,
        tvec);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    cv::Mat rotation32f;
    rotation_matrix.convertTo(rotation32f, CV_32F);
    cv::Matx33f rotation_cam0_from_world = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotation_cam0_from_world(r, c) = rotation32f.at<float>(r, c);
        }
    }
    const cv::Matx33f rotation_world_from_cam0 = rotation_cam0_from_world.t();
    const cv::Vec3f translation_cam0_from_world = ToVec3f(tvec);

    result.detected = true;
    result.rvec = rvec;
    result.tvec = tvec;
    result.reprojection_error_px = MeanCornerReprojectionError(
        object_points,
        image_points,
        rvec,
        tvec,
        undistorted_camera_matrix,
        undistorted_zero_dist_coeffs);
    return {
        true,
        result.num_charuco_corners,
        result.reprojection_error_px,
        rvec,
        tvec,
        result.charuco_corners,
        result.charuco_ids,
        result.marker_ids,
        result.marker_corners
    };
}

newnewhand::StereoCameraTrackingResult UpdateCharucoTrackingResult(
    const CharucoDetectionResult& detection,
    const newnewhand::StereoSingleViewPoseFrame& stereo_frame,
    CharucoLocalizationState& state) {
    newnewhand::StereoCameraTrackingResult tracking_result;
    tracking_result.capture_index = stereo_frame.capture_index;
    tracking_result.trigger_timestamp = stereo_frame.trigger_timestamp;
    tracking_result.left_keypoints = detection.num_charuco_corners;
    if (!detection.detected) {
        tracking_result.status_message = state.has_pose
            ? "charuco board lost, holding last pose"
            : "charuco board not detected";
        if (state.has_pose) {
            tracking_result.initialized = true;
            tracking_result.rotation_world_from_cam0 = state.rotation_world_from_cam0;
            tracking_result.camera_center_world = state.camera_center_world;
            tracking_result.trajectory_world = state.trajectory_world;
        }
        return tracking_result;
    }

    cv::Mat rotation_matrix;
    cv::Rodrigues(detection.rvec, rotation_matrix);
    cv::Mat rotation32f;
    rotation_matrix.convertTo(rotation32f, CV_32F);
    cv::Matx33f rotation_cam0_from_world = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotation_cam0_from_world(r, c) = rotation32f.at<float>(r, c);
        }
    }
    const cv::Matx33f rotation_world_from_cam0 = rotation_cam0_from_world.t();
    const cv::Vec3f translation_cam0_from_world = ToVec3f(detection.tvec);
    const cv::Vec3f camera_center_world =
        -MatVecMul(rotation_world_from_cam0, translation_cam0_from_world);

    state.has_pose = true;
    state.rotation_world_from_cam0 = rotation_world_from_cam0;
    state.camera_center_world = camera_center_world;
    state.trajectory_world.push_back(camera_center_world);

    tracking_result.initialized = true;
    tracking_result.tracking_ok = true;
    tracking_result.rotation_world_from_cam0 = state.rotation_world_from_cam0;
    tracking_result.camera_center_world = state.camera_center_world;
    tracking_result.trajectory_world = state.trajectory_world;
    tracking_result.status_message = "charuco board detected in left camera";
    return tracking_result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);

        newnewhand::StereoSingleViewHandPosePipelineConfig pipeline_config;
        pipeline_config.capture_config.camera_settings.exposure_us = options.exposure_us;
        pipeline_config.capture_config.camera_settings.gain = options.gain;
        pipeline_config.pose_config.detector_model_path = options.detector_model_path;
        pipeline_config.pose_config.wilor_model_path = options.wilor_model_path;
        pipeline_config.pose_config.mano_model_path = options.mano_model_path;
        pipeline_config.pose_config.debug_dump_dir = options.debug_dir;
        pipeline_config.pose_config.ort_profile_prefix = options.ort_profile_prefix;
        pipeline_config.pose_config.use_gpu = options.use_gpu;

        const auto calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        if (options.slam && options.charuco_localization) {
            throw std::runtime_error("use at most one of --slam and --charuco_localization");
        }
        if (options.charuco_localization) {
            if (options.squares_x < 3 || options.squares_y < 3) {
                throw std::runtime_error("charuco localization requires at least a 3x3 board");
            }
            if (options.square_length_m <= 0.0f || options.marker_length_m <= 0.0f) {
                throw std::runtime_error("--square_length_m and --marker_length_m must be positive");
            }
            if (options.marker_length_m >= options.square_length_m) {
                throw std::runtime_error("--marker_length_m must be smaller than --square_length_m");
            }
        }
        ApplyCalibrationSerialsToCaptureConfig(options, calibration, pipeline_config);
        newnewhand::StereoHandFuserConfig fuser_config;
        fuser_config.calibration = calibration;
        fuser_config.require_both_views = true;
        fuser_config.verbose_logging = options.verbose;

        newnewhand::GlfwSceneViewerConfig viewer_config;
        viewer_config.has_cam1_pose = true;
        viewer_config.cam1_rotation_cam1_to_cam0 = cv::Matx33f(calibration.rotation).t();
        const cv::Vec3f translation_01(calibration.translation);
        viewer_config.cam1_center_cam0 = -(viewer_config.cam1_rotation_cam1_to_cam0 * translation_01);
        newnewhand::GlfwSceneViewer viewer(viewer_config);

        if (options.glfw_view && !viewer.Initialize()) {
            throw std::runtime_error("failed to initialize GLFW OpenGL viewer");
        }

        const auto frame_interval = options.fps == 0
            ? std::chrono::microseconds(0)
            : std::chrono::microseconds(1000000 / options.fps);

        SharedWorkerState shared_state;
        std::atomic<bool> stop_requested = false;
        std::atomic<bool> worker_finished = false;
        std::thread worker([&]() {
            try {
                newnewhand::StereoHandFuser fuser(std::move(fuser_config));
                std::unique_ptr<newnewhand::OfflineSequenceWriter> offline_writer;
                if (!options.offline_dump_dir.empty()) {
                    newnewhand::OfflineSequenceWriterConfig writer_config;
                    writer_config.output_root = options.offline_dump_dir;
                    writer_config.calibration_source_path = options.calibration_path;
                    writer_config.save_raw_images = true;
                    writer_config.save_overlay_images = true;
                    offline_writer = std::make_unique<newnewhand::OfflineSequenceWriter>(writer_config, calibration);
                    offline_writer->Initialize();
                }

                newnewhand::StereoSingleViewHandPosePipeline pipeline(pipeline_config);
                std::unique_ptr<newnewhand::StereoVisualOdometry> slam_tracker;
                cv::Mat undistort_left_map_x;
                cv::Mat undistort_left_map_y;
                cv::Mat undistort_left_camera_matrix;
                cv::Mat undistort_zero_dist_coeffs;
                std::unique_ptr<cv::aruco::CharucoBoard> charuco_board;
                std::unique_ptr<cv::aruco::CharucoDetector> charuco_detector;
                CharucoLocalizationState charuco_state;
                if (options.slam) {
                    newnewhand::StereoVisualOdometryConfig slam_config;
                    slam_config.calibration = calibration;
                    slam_config.verbose_logging = options.verbose;
                    slam_tracker = std::make_unique<newnewhand::StereoVisualOdometry>(std::move(slam_config));
                } else if (options.charuco_localization) {
                    undistort_left_camera_matrix = BuildUndistortedCameraMatrix(
                        calibration.left_camera_matrix,
                        calibration.left_dist_coeffs,
                        calibration.image_size);
                    undistort_zero_dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
                    cv::initUndistortRectifyMap(
                        calibration.left_camera_matrix,
                        calibration.left_dist_coeffs,
                        cv::Mat(),
                        undistort_left_camera_matrix,
                        calibration.image_size,
                        CV_32FC1,
                        undistort_left_map_x,
                        undistort_left_map_y);
                    const cv::aruco::Dictionary dictionary =
                        cv::aruco::getPredefinedDictionary(ParseDictionaryName(options.dictionary_name));
                    charuco_board = std::make_unique<cv::aruco::CharucoBoard>(
                        cv::Size(options.squares_x, options.squares_y),
                        options.square_length_m,
                        options.marker_length_m,
                        dictionary);
                    charuco_board->setLegacyPattern(options.legacy_pattern);
                    cv::aruco::CharucoParameters charuco_parameters;
                    charuco_parameters.cameraMatrix = undistort_left_camera_matrix;
                    charuco_parameters.distCoeffs = undistort_zero_dist_coeffs;
                    cv::aruco::DetectorParameters detector_parameters;
                    detector_parameters.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
                    charuco_detector = std::make_unique<cv::aruco::CharucoDetector>(
                        *charuco_board,
                        charuco_parameters,
                        detector_parameters);
                }
                pipeline.Initialize();
                pipeline.Start();
                ValidateActiveCamerasAgainstCalibration(pipeline.ActiveCameras(), calibration);

                try {
                    for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
                        if (stop_requested.load()) {
                            break;
                        }

                        const auto loop_start = std::chrono::steady_clock::now();
                        auto stereo_frame = pipeline.CaptureAndEstimate();
                        if (stop_requested.load()) {
                            break;
                        }

                        if (options.verbose) {
                            std::cerr << "[app] capture=" << stereo_frame.capture_index << "\n";
                            for (std::size_t view_index = 0; view_index < stereo_frame.views.size(); ++view_index) {
                                const auto& view = stereo_frame.views[view_index];
                                std::cerr
                                    << "[app] view" << view_index
                                    << " hands=" << view.hand_poses.size()
                                    << " valid=" << view.camera_frame.valid
                                    << " error=" << (view.inference_error.empty() ? "<none>" : view.inference_error)
                                    << "\n";
                                for (std::size_t hand_index = 0; hand_index < view.hand_poses.size(); ++hand_index) {
                                    const auto& hand = view.hand_poses[hand_index];
                                    std::cerr
                                        << "[app] view" << view_index
                                        << " hand" << hand_index
                                        << " type=" << (hand.detection.is_right ? "right" : "left")
                                        << " bbox=(" << hand.detection.bbox[0] << ", " << hand.detection.bbox[1]
                                        << ", " << hand.detection.bbox[2] << ", " << hand.detection.bbox[3] << ")"
                                        << " root2d=" << FormatPoint(hand.keypoints_2d[0][0], hand.keypoints_2d[0][1])
                                        << "\n";
                                }
                            }
                        }

                        auto fused_frame = fuser.Fuse(stereo_frame);
                        newnewhand::StereoCameraTrackingResult tracking_result;
                        CharucoDetectionResult charuco_detection;
                        if (slam_tracker) {
                            tracking_result = slam_tracker->Track(stereo_frame);
                        } else if (charuco_detector) {
                            cv::Mat undistorted_left_image;
                            cv::remap(
                                stereo_frame.views[0].camera_frame.bgr_image,
                                undistorted_left_image,
                                undistort_left_map_x,
                                undistort_left_map_y,
                                cv::INTER_LINEAR);
                            charuco_detection = EstimateCharucoPoseInLeftCamera(
                                undistorted_left_image,
                                *charuco_board,
                                *charuco_detector,
                                undistort_left_camera_matrix,
                                undistort_zero_dist_coeffs);
                            tracking_result = UpdateCharucoTrackingResult(
                                charuco_detection,
                                stereo_frame,
                                charuco_state);
                        }
                        std::cout
                            << "capture=" << fused_frame.capture_index
                            << " fused_hands=" << fused_frame.hands.size() << "\n";

                        auto latest_frame = std::make_shared<LatestFrameData>();
                        latest_frame->stereo_frame = stereo_frame;
                        latest_frame->fused_frame = fused_frame;
                        latest_frame->tracking_result = tracking_result;
                        latest_frame->charuco_detection = std::move(charuco_detection);
                        {
                            std::lock_guard<std::mutex> lock(shared_state.mutex);
                            shared_state.latest_frame = std::move(latest_frame);
                        }

                        if (options.save) {
                            for (std::size_t i = 0; i < stereo_frame.views.size(); ++i) {
                                if (!stereo_frame.views[i].overlay_image.empty()) {
                                    SaveImage(
                                        options.output_dir,
                                        "cam" + std::to_string(i),
                                        fused_frame.capture_index,
                                        stereo_frame.views[i].overlay_image);
                                }
                            }
                            SaveFusedYaml(fuser, fused_frame, options.output_dir);
                        }

                        if (offline_writer) {
                            offline_writer->SaveFrame(stereo_frame, fused_frame);
                        }

                        if (frame_interval.count() > 0) {
                            const auto elapsed = std::chrono::steady_clock::now() - loop_start;
                            const auto remaining =
                                frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
                            if (remaining.count() > 0) {
                                std::this_thread::sleep_for(remaining);
                            }
                        }
                    }
                } catch (...) {
                    pipeline.Stop();
                    throw;
                }

                pipeline.Stop();
            } catch (...) {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                shared_state.worker_error = std::current_exception();
            }
            worker_finished.store(true);
        });

        std::shared_ptr<const LatestFrameData> latest_frame;
        std::exception_ptr worker_error;
        const newnewhand::StereoFusedHandPoseFrame empty_fused_frame;

        while (true) {
            {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                if (shared_state.latest_frame) {
                    latest_frame = shared_state.latest_frame;
                }
                if (shared_state.worker_error) {
                    worker_error = shared_state.worker_error;
                }
            }

            if (worker_error) {
                break;
            }

            if (options.glfw_view) {
                const auto& fused_frame = latest_frame ? latest_frame->fused_frame : empty_fused_frame;
                const newnewhand::StereoCameraTrackingResult* tracking_result =
                    latest_frame ? &latest_frame->tracking_result : nullptr;
                if (!viewer.Render(fused_frame, tracking_result)) {
                    std::cerr << "[app] GLFW viewer closed by user\n";
                    stop_requested.store(true);
                    break;
                }
            }

            if (options.preview) {
                if (latest_frame) {
                    for (std::size_t i = 0; i < latest_frame->stereo_frame.views.size(); ++i) {
                        if (!latest_frame->stereo_frame.views[i].overlay_image.empty()) {
                            cv::Mat preview;
                            cv::resize(
                                latest_frame->stereo_frame.views[i].overlay_image,
                                preview,
                                cv::Size(),
                                0.5,
                                0.5);
                            std::ostringstream tracking_text;
                            if (options.charuco_localization) {
                                tracking_text
                                    << "charuco "
                                    << (latest_frame->tracking_result.initialized
                                            ? (latest_frame->tracking_result.tracking_ok ? "ok" : "hold")
                                            : "init")
                                    << " corners=" << latest_frame->charuco_detection.num_charuco_corners
                                    << " reproj=" << std::fixed << std::setprecision(3)
                                    << latest_frame->charuco_detection.reprojection_error_px;
                            } else {
                                tracking_text
                                    << "slam "
                                    << (latest_frame->tracking_result.initialized
                                            ? (latest_frame->tracking_result.tracking_ok ? "ok" : "hold")
                                            : "init")
                                    << " kp=" << latest_frame->tracking_result.left_keypoints
                                    << " stereo=" << latest_frame->tracking_result.stereo_points
                                    << " match=" << latest_frame->tracking_result.matched_points
                                    << " inlier=" << latest_frame->tracking_result.tracking_inliers;
                            }
                            cv::putText(
                                preview,
                                tracking_text.str(),
                                cv::Point(16, 28),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.55,
                                cv::Scalar(40, 220, 255),
                                2);
                            if (!latest_frame->tracking_result.status_message.empty()) {
                                std::ostringstream tracking_detail_text;
                                if (options.charuco_localization) {
                                    tracking_detail_text
                                        << "dict=" << options.dictionary_name
                                        << " board=" << options.squares_x << "x" << options.squares_y;
                                } else {
                                    tracking_detail_text
                                        << "disp_kp=" << latest_frame->tracking_result.valid_disparity_keypoints
                                        << " nan=" << latest_frame->tracking_result.invalid_nonfinite_disparity
                                        << " low=" << latest_frame->tracking_result.invalid_low_disparity
                                        << " depth=" << latest_frame->tracking_result.invalid_depth;
                                }
                                cv::putText(
                                    preview,
                                    tracking_detail_text.str(),
                                    cv::Point(16, 54),
                                    cv::FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    cv::Scalar(40, 220, 255),
                                    1);
                                cv::putText(
                                    preview,
                                    latest_frame->tracking_result.status_message,
                                    cv::Point(16, 78),
                                    cv::FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    cv::Scalar(40, 220, 255),
                                    1);
                            }
                            if (latest_frame->tracking_result.initialized) {
                                std::ostringstream slam_pose_text;
                                slam_pose_text
                                    << "xyz=("
                                    << latest_frame->tracking_result.camera_center_world[0] << ", "
                                    << latest_frame->tracking_result.camera_center_world[1] << ", "
                                    << latest_frame->tracking_result.camera_center_world[2] << ")";
                                cv::putText(
                                    preview,
                                    slam_pose_text.str(),
                                    cv::Point(16, 102),
                                    cv::FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    cv::Scalar(40, 220, 255),
                                    1);
                            }
                            cv::imshow("fused_pose_cam" + std::to_string(i), preview);
                        }
                    }
                }
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    stop_requested.store(true);
                    break;
                }
            }

            if (worker_finished.load()) {
                break;
            }

            if (!options.glfw_view) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        stop_requested.store(true);
        if (worker.joinable()) {
            worker.join();
        }
        if (worker_error) {
            std::rethrow_exception(worker_error);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
