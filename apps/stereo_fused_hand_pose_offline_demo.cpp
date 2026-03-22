#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/fusion/stereo_hand_fuser.h"
#include "newnewhand/io/offline_sequence_writer.h"
#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"
#include "newnewhand/render/glfw_scene_viewer.h"
#include "newnewhand/slam/offline_camera_trajectory_optimizer.h"
#include "newnewhand/slam/stereo_visual_odometry.h"
#ifdef NEWNEWHAND_HAVE_STELLA_VSLAM
#include "newnewhand/slam/stella_vslam_tracker.h"
#endif
#include "newnewhand/types/stereo_frame.h"

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

struct DemoOptions {
    std::filesystem::path input_dir;
    std::string calibration_path;
    std::string output_dir = "results/stereo_fused_hand_pose_offline";
    std::string offline_dump_dir;
    std::string debug_dir = "debug/wilor_failures";
    std::string ort_profile_prefix;
    std::string slam_backend = "stella";
    std::string stella_vocab_path;
    std::string stella_config_dump_path;
    std::string detector_model_path = DefaultDetectorModelPath();
    std::string wilor_model_path = DefaultWilorModelPath();
    std::string mano_model_path = DefaultManoModelPath();
    unsigned int playback_fps = 0;
    int frames = -1;
    bool preview = true;
    bool save = true;
    bool use_gpu = true;
    bool verbose = true;
    bool glfw_view = true;
    std::string dictionary_name = "DICT_APRILTAG_36h11";
    int squares_x = 5;
    int squares_y = 7;
    float square_length_m = 0.028f;
    float marker_length_m = 0.021f;
    bool legacy_pattern = false;
};

struct BoardDetectionResult {
    bool detected = false;
    int num_charuco_corners = 0;
    float reprojection_error_px = 0.0f;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat charuco_corners;
    cv::Mat charuco_ids;
    cv::Mat marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct PreviousPoseState {
    bool initialized = false;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Vec3f board_normal_in_cam = cv::Vec3f(0.0f, 0.0f, 1.0f);
};

struct SlamAlignmentState {
    bool has_alignment = false;
    bool seeded_from_last_calibrated_pose = false;
    std::uint64_t reference_capture_index = 0;
    cv::Matx33f rotation_world_from_slam = cv::Matx33f::eye();
    cv::Vec3f translation_world_from_slam = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct LastCalibratedPoseState {
    bool has_pose = false;
    std::uint64_t capture_index = 0;
    cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct HybridTrackingState {
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

        if (arg == "--input_dir") options.input_dir = require_value(arg);
        else if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--output_dir") options.output_dir = require_value(arg);
        else if (arg == "--offline_dump_dir") options.offline_dump_dir = require_value(arg);
        else if (arg == "--debug_dir") options.debug_dir = require_value(arg);
        else if (arg == "--ort_profile") options.ort_profile_prefix = require_value(arg);
        else if (arg == "--slam_backend") options.slam_backend = require_value(arg);
        else if (arg == "--stella_vocab") options.stella_vocab_path = require_value(arg);
        else if (arg == "--stella_config_dump") options.stella_config_dump_path = require_value(arg);
        else if (arg == "--detector_model") options.detector_model_path = require_value(arg);
        else if (arg == "--wilor_model") options.wilor_model_path = require_value(arg);
        else if (arg == "--mano_model") options.mano_model_path = require_value(arg);
        else if (arg == "--fps") options.playback_fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--save") options.save = true;
        else if (arg == "--no_save") options.save = false;
        else if (arg == "--glfw_view") options.glfw_view = true;
        else if (arg == "--no_glfw_view") options.glfw_view = false;
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
                << "Usage: stereo_fused_hand_pose_offline_demo [options]\n"
                << "  --input_dir <dir>         required\n"
                << "  --calibration <path>      default: <input_dir>/calibration/stereo_calibration.yaml or "
                << DefaultCalibrationPath() << "\n"
                << "  --detector_model <path>   default: " << DefaultDetectorModelPath() << "\n"
                << "  --wilor_model <path>      default: " << DefaultWilorModelPath() << "\n"
                << "  --mano_model <path>       default: " << DefaultManoModelPath() << "\n"
                << "  --output_dir <dir>        default: results/stereo_fused_hand_pose_offline\n"
                << "  --offline_dump_dir <dir>  default: disabled\n"
                << "  --slam_backend <name>     default: stella (stella|legacy)\n"
                << "  --stella_vocab <path>     default: disabled\n"
                << "  --stella_config_dump <path>  default: disabled\n"
                << "  --fps <int>               default: 0 (process as fast as possible)\n"
                << "  --frames <int>            default: -1 (use all pairs)\n"
                << "  --save | --no_save        default: --save\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
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

    if (options.input_dir.empty()) {
        throw std::runtime_error("--input_dir is required");
    }
    if (options.slam_backend != "stella" && options.slam_backend != "legacy") {
        throw std::runtime_error("--slam_backend must be stella or legacy");
    }
    if (options.squares_x < 3 || options.squares_y < 3) {
        throw std::runtime_error("board localization requires at least a 3x3 ChArUco board");
    }
    if (options.square_length_m <= 0.0f || options.marker_length_m <= 0.0f) {
        throw std::runtime_error("--square_length_m and --marker_length_m must be positive");
    }
    if (options.marker_length_m >= options.square_length_m) {
        throw std::runtime_error("--marker_length_m must be smaller than --square_length_m");
    }
    return options;
}

std::string ResolveCalibrationPath(const DemoOptions& options) {
    if (!options.calibration_path.empty()) {
        return options.calibration_path;
    }

    const std::filesystem::path package_calibration =
        options.input_dir / "calibration" / "stereo_calibration.yaml";
    if (std::filesystem::exists(package_calibration)) {
        return package_calibration.string();
    }
    return DefaultCalibrationPath();
}

std::pair<std::filesystem::path, std::filesystem::path> ResolveInputImageDirs(
    const std::filesystem::path& input_dir) {
    const std::filesystem::path packaged_left = input_dir / "images" / "cam0";
    const std::filesystem::path packaged_right = input_dir / "images" / "cam1";
    if (std::filesystem::is_directory(packaged_left) && std::filesystem::is_directory(packaged_right)) {
        return {packaged_left, packaged_right};
    }

    const std::filesystem::path flat_left = input_dir / "cam0";
    const std::filesystem::path flat_right = input_dir / "cam1";
    if (std::filesystem::is_directory(flat_left) && std::filesystem::is_directory(flat_right)) {
        return {flat_left, flat_right};
    }

    throw std::runtime_error(
        "input_dir must contain either images/cam0 + images/cam1 or cam0 + cam1");
}

std::uint64_t ParseCaptureIndexFromPath(const std::filesystem::path& path, std::uint64_t fallback_index) {
    const std::string stem = path.stem().string();
    if (stem.empty()) {
        return fallback_index;
    }
    for (char ch : stem) {
        if (ch < '0' || ch > '9') {
            return fallback_index;
        }
    }
    return static_cast<std::uint64_t>(std::stoull(stem));
}

newnewhand::StereoSingleViewPoseFrame ToTrackingFrame(const newnewhand::StereoFrame& stereo_frame) {
    newnewhand::StereoSingleViewPoseFrame tracking_frame;
    tracking_frame.capture_index = stereo_frame.capture_index;
    tracking_frame.trigger_timestamp = stereo_frame.trigger_timestamp;
    for (std::size_t i = 0; i < stereo_frame.views.size(); ++i) {
        tracking_frame.views[i].camera_frame = stereo_frame.views[i];
    }
    return tracking_frame;
}

newnewhand::StereoFrame LoadStereoFrame(
    const newnewhand::CalibrationImagePair& pair,
    std::uint64_t capture_index,
    const newnewhand::StereoCalibrationResult& calibration) {
    newnewhand::StereoFrame stereo_frame;
    stereo_frame.capture_index = capture_index;
    stereo_frame.trigger_timestamp = std::chrono::steady_clock::now();

    const std::array<std::filesystem::path, 2> paths = {pair.left_path, pair.right_path};
    const std::array<std::string, 2> serials = {
        calibration.left_camera_serial_number,
        calibration.right_camera_serial_number,
    };

    for (std::size_t i = 0; i < paths.size(); ++i) {
        auto& frame = stereo_frame.views[i];
        frame.camera_index = i;
        frame.serial_number = serials[i];
        frame.bgr_image = cv::imread(paths[i].string(), cv::IMREAD_COLOR);
        frame.valid = !frame.bgr_image.empty();
        frame.frame_index = capture_index;
        if (!frame.valid) {
            frame.error_message = "failed to read image: " + paths[i].string();
            continue;
        }
        if (calibration.image_size.width > 0 && frame.bgr_image.size() != calibration.image_size) {
            std::ostringstream oss;
            oss << "image size mismatch for " << paths[i]
                << ": expected " << calibration.image_size.width << "x" << calibration.image_size.height
                << " got " << frame.bgr_image.cols << "x" << frame.bgr_image.rows;
            throw std::runtime_error(oss.str());
        }
        frame.width = static_cast<std::uint32_t>(frame.bgr_image.cols);
        frame.height = static_cast<std::uint32_t>(frame.bgr_image.rows);
    }

    return stereo_frame;
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

newnewhand::StereoFrame UndistortStereoFrame(
    const newnewhand::StereoFrame& raw_frame,
    const cv::Mat& left_map_x,
    const cv::Mat& left_map_y,
    const cv::Mat& right_map_x,
    const cv::Mat& right_map_y) {
    newnewhand::StereoFrame undistorted_frame = raw_frame;
    if (!raw_frame.views[0].bgr_image.empty()) {
        cv::remap(
            raw_frame.views[0].bgr_image,
            undistorted_frame.views[0].bgr_image,
            left_map_x,
            left_map_y,
            cv::INTER_LINEAR);
    }
    if (!raw_frame.views[1].bgr_image.empty()) {
        cv::remap(
            raw_frame.views[1].bgr_image,
            undistorted_frame.views[1].bgr_image,
            right_map_x,
            right_map_y,
            cv::INTER_LINEAR);
    }
    return undistorted_frame;
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

cv::Vec3f BoardNormalInCamera(const cv::Mat& rvec) {
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    cv::Mat rotation32f;
    rotation_matrix.convertTo(rotation32f, CV_32F);
    return cv::Vec3f(
        rotation32f.at<float>(0, 2),
        rotation32f.at<float>(1, 2),
        rotation32f.at<float>(2, 2));
}

float CosineSimilarity(const cv::Vec3f& a, const cv::Vec3f& b) {
    const float denom = cv::norm(a) * cv::norm(b);
    if (denom <= 1e-8f) {
        return 1.0f;
    }
    return a.dot(b) / denom;
}

BoardDetectionResult EstimateBoardPoseInLeftCamera(
    const cv::Mat& undistorted_left_image,
    const cv::aruco::CharucoBoard& board,
    const cv::aruco::CharucoDetector& detector,
    const cv::Mat& undistorted_camera_matrix,
    const cv::Mat& undistorted_zero_dist_coeffs,
    PreviousPoseState* previous_pose_state) {
    BoardDetectionResult result;

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

    cv::Mat rvec = previous_pose_state && previous_pose_state->initialized
        ? previous_pose_state->rvec.clone()
        : cv::Mat();
    cv::Mat tvec = previous_pose_state && previous_pose_state->initialized
        ? previous_pose_state->tvec.clone()
        : cv::Mat();
    const bool pnp_ok = cv::solvePnP(
        object_points,
        image_points,
        undistorted_camera_matrix,
        undistorted_zero_dist_coeffs,
        rvec,
        tvec,
        previous_pose_state && previous_pose_state->initialized,
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

    const cv::Vec3f current_board_normal_in_cam = BoardNormalInCamera(rvec);
    if (previous_pose_state && previous_pose_state->initialized) {
        const float normal_cos =
            CosineSimilarity(current_board_normal_in_cam, previous_pose_state->board_normal_in_cam);
        if (normal_cos < -0.2f) {
            return result;
        }
    }

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
    result.rotation_world_from_cam0 = rotation_world_from_cam0;
    result.camera_center_world = -MatVecMul(rotation_world_from_cam0, translation_cam0_from_world);

    if (previous_pose_state) {
        previous_pose_state->initialized = true;
        previous_pose_state->rvec = rvec.clone();
        previous_pose_state->tvec = tvec.clone();
        previous_pose_state->board_normal_in_cam = current_board_normal_in_cam;
    }
    return result;
}

void UpdateSlamAlignmentFromPose(
    const cv::Matx33f& rotation_world_from_cam0,
    const cv::Vec3f& camera_center_world,
    const newnewhand::StereoCameraTrackingResult& slam_result,
    SlamAlignmentState& alignment_state,
    bool seeded_from_last_calibrated_pose,
    std::uint64_t reference_capture_index) {
    if (!slam_result.initialized) {
        return;
    }
    alignment_state.rotation_world_from_slam =
        rotation_world_from_cam0 * slam_result.rotation_world_from_cam0.t();
    alignment_state.translation_world_from_slam =
        camera_center_world - MatVecMul(alignment_state.rotation_world_from_slam, slam_result.camera_center_world);
    alignment_state.has_alignment = true;
    alignment_state.seeded_from_last_calibrated_pose = seeded_from_last_calibrated_pose;
    alignment_state.reference_capture_index = reference_capture_index;
}

newnewhand::StereoCameraTrackingResult TransformAlignedSlamResult(
    const newnewhand::StereoCameraTrackingResult& slam_result,
    const SlamAlignmentState& alignment_state) {
    newnewhand::StereoCameraTrackingResult result = slam_result;
    result.rotation_world_from_cam0 =
        alignment_state.rotation_world_from_slam * slam_result.rotation_world_from_cam0;
    result.camera_center_world =
        MatVecMul(alignment_state.rotation_world_from_slam, slam_result.camera_center_world)
        + alignment_state.translation_world_from_slam;
    result.trajectory_world.clear();
    result.trajectory_world.reserve(slam_result.trajectory_world.size());
    for (const auto& center_slam : slam_result.trajectory_world) {
        result.trajectory_world.push_back(
            MatVecMul(alignment_state.rotation_world_from_slam, center_slam)
            + alignment_state.translation_world_from_slam);
    }
    return result;
}

newnewhand::StereoCameraTrackingResult ResolveHybridTrackingResult(
    const BoardDetectionResult& board_detection,
    const newnewhand::StereoCameraTrackingResult& slam_result,
    std::uint64_t capture_index,
    std::chrono::steady_clock::time_point trigger_timestamp,
    SlamAlignmentState& slam_alignment_state,
    LastCalibratedPoseState& last_calibrated_pose_state,
    HybridTrackingState& hybrid_state) {
    if (board_detection.detected) {
        last_calibrated_pose_state.has_pose = true;
        last_calibrated_pose_state.capture_index = capture_index;
        last_calibrated_pose_state.rotation_world_from_cam0 = board_detection.rotation_world_from_cam0;
        last_calibrated_pose_state.camera_center_world = board_detection.camera_center_world;
    }

    if (slam_result.reinitialized) {
        slam_alignment_state.has_alignment = false;
    }
    if (board_detection.detected) {
        UpdateSlamAlignmentFromPose(
            board_detection.rotation_world_from_cam0,
            board_detection.camera_center_world,
            slam_result,
            slam_alignment_state,
            false,
            capture_index);
    } else if (!slam_alignment_state.has_alignment && last_calibrated_pose_state.has_pose && slam_result.initialized) {
        UpdateSlamAlignmentFromPose(
            last_calibrated_pose_state.rotation_world_from_cam0,
            last_calibrated_pose_state.camera_center_world,
            slam_result,
            slam_alignment_state,
            true,
            last_calibrated_pose_state.capture_index);
    }

    newnewhand::StereoCameraTrackingResult result;
    result.capture_index = capture_index;
    result.trigger_timestamp = trigger_timestamp;
    result.left_keypoints = board_detection.detected ? board_detection.num_charuco_corners : slam_result.left_keypoints;
    result.stereo_points = slam_result.stereo_points;
    result.valid_disparity_keypoints = slam_result.valid_disparity_keypoints;
    result.invalid_nonfinite_disparity = slam_result.invalid_nonfinite_disparity;
    result.invalid_low_disparity = slam_result.invalid_low_disparity;
    result.invalid_depth = slam_result.invalid_depth;
    result.matched_points = slam_result.matched_points;
    result.tracking_inliers = slam_result.tracking_inliers;

    bool append_pose = false;
    if (board_detection.detected) {
        result.initialized = true;
        result.tracking_ok = true;
        result.rotation_world_from_cam0 = board_detection.rotation_world_from_cam0;
        result.camera_center_world = board_detection.camera_center_world;
        result.status_message = "charuco board detected in left camera";
        append_pose = true;
    } else if (slam_alignment_state.has_alignment && slam_result.initialized) {
        result = TransformAlignedSlamResult(slam_result, slam_alignment_state);
        result.capture_index = capture_index;
        result.trigger_timestamp = trigger_timestamp;
        if (slam_alignment_state.seeded_from_last_calibrated_pose) {
            result.status_message = slam_result.tracking_ok
                ? "charuco missing, using slam fallback seeded from last calibrated pose"
                : "charuco missing, holding aligned slam pose seeded from last calibrated pose";
        } else {
            result.status_message = slam_result.tracking_ok
                ? "charuco missing, using slam fallback"
                : "charuco missing, holding aligned slam pose";
        }
        append_pose = slam_result.tracking_ok;
    } else if (hybrid_state.has_pose) {
        result.initialized = true;
        result.tracking_ok = false;
        result.rotation_world_from_cam0 = hybrid_state.rotation_world_from_cam0;
        result.camera_center_world = hybrid_state.camera_center_world;
        result.status_message = "charuco missing and slam fallback unavailable, holding last pose";
    } else {
        result.status_message = "charuco missing and slam fallback unavailable";
    }

    if (result.initialized) {
        if (append_pose) {
            hybrid_state.has_pose = true;
            hybrid_state.rotation_world_from_cam0 = result.rotation_world_from_cam0;
            hybrid_state.camera_center_world = result.camera_center_world;
            hybrid_state.trajectory_world.push_back(result.camera_center_world);
        }
        result.trajectory_world = hybrid_state.trajectory_world;
    }

    return result;
}

void DrawTrackingOverlay(
    cv::Mat& image,
    const std::string& label,
    const BoardDetectionResult& board_detection,
    const newnewhand::StereoCameraTrackingResult& tracking_result,
    double processing_fps) {
    cv::putText(
        image,
        label,
        cv::Point(16, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(40, 220, 255),
        2);

    std::ostringstream source_text;
    source_text
        << "loc="
        << (board_detection.detected ? "charuco" : (tracking_result.status_message.find("slam") != std::string::npos ? "slam" : "hold"))
        << " fps=" << std::fixed << std::setprecision(1) << processing_fps
        << " corners=" << board_detection.num_charuco_corners
        << " reproj=" << std::fixed << std::setprecision(3) << board_detection.reprojection_error_px;
    cv::putText(
        image,
        source_text.str(),
        cv::Point(16, 54),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(40, 220, 255),
        1);

    std::ostringstream slam_text;
    slam_text
        << "slam kp=" << tracking_result.left_keypoints
        << " stereo=" << tracking_result.stereo_points
        << " match=" << tracking_result.matched_points
        << " inlier=" << tracking_result.tracking_inliers;
    cv::putText(
        image,
        slam_text.str(),
        cv::Point(16, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(40, 220, 255),
        1);

    if (!tracking_result.status_message.empty()) {
        cv::putText(
            image,
            tracking_result.status_message,
            cv::Point(16, 104),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(40, 220, 255),
            1);
    }

    if (tracking_result.initialized) {
        std::ostringstream pose_text;
        pose_text
            << "xyz=("
            << tracking_result.camera_center_world[0] << ", "
            << tracking_result.camera_center_world[1] << ", "
            << tracking_result.camera_center_world[2] << ")";
        cv::putText(
            image,
            pose_text.str(),
            cv::Point(16, 128),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(40, 220, 255),
            1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);
        const std::string calibration_path = ResolveCalibrationPath(options);
        const auto calibration = newnewhand::StereoCalibrator::LoadResult(calibration_path);
        const auto [left_dir, right_dir] = ResolveInputImageDirs(options.input_dir);

        auto image_pairs = newnewhand::StereoCalibrator::CollectImagePairs(left_dir, right_dir);
        if (image_pairs.empty()) {
            throw std::runtime_error("no matched offline stereo image pairs found");
        }
        if (options.frames > 0 && static_cast<std::size_t>(options.frames) < image_pairs.size()) {
            image_pairs.resize(static_cast<std::size_t>(options.frames));
        }

        newnewhand::StereoSingleViewHandPosePipelineConfig pipeline_config;
        pipeline_config.pose_config.detector_model_path = options.detector_model_path;
        pipeline_config.pose_config.wilor_model_path = options.wilor_model_path;
        pipeline_config.pose_config.mano_model_path = options.mano_model_path;
        pipeline_config.pose_config.debug_dump_dir = options.debug_dir;
        pipeline_config.pose_config.ort_profile_prefix = options.ort_profile_prefix;
        pipeline_config.pose_config.use_gpu = options.use_gpu;
        newnewhand::StereoSingleViewHandPosePipeline pose_pipeline(pipeline_config);

        newnewhand::StereoHandFuserConfig fuser_config;
        fuser_config.calibration = calibration;
        fuser_config.require_both_views = true;
        fuser_config.verbose_logging = options.verbose;
        fuser_config.input_views_are_undistorted = true;
        newnewhand::StereoHandFuser fuser(fuser_config);

        const bool use_stella_backend = options.slam_backend == "stella";
#ifndef NEWNEWHAND_HAVE_STELLA_VSLAM
        if (use_stella_backend) {
            throw std::runtime_error(
                "this build does not include stella_vslam; reconfigure with -DNEWHAND_ENABLE_STELLA_VSLAM=ON");
        }
#endif

        std::unique_ptr<newnewhand::StereoVisualOdometry> legacy_slam_tracker;
#ifdef NEWNEWHAND_HAVE_STELLA_VSLAM
        std::unique_ptr<newnewhand::StellaStereoSlamTracker> stella_slam_tracker;
#endif

        if (use_stella_backend) {
#ifdef NEWNEWHAND_HAVE_STELLA_VSLAM
            newnewhand::StellaStereoSlamTrackerConfig stella_config;
            stella_config.calibration = calibration;
            stella_config.vocab_path = options.stella_vocab_path;
            stella_config.nominal_fps = options.playback_fps > 0 ? static_cast<double>(options.playback_fps) : 30.0;
            stella_config.verbose_logging = options.verbose;
            stella_config.generated_config_dump_path = options.stella_config_dump_path;
            stella_slam_tracker = std::make_unique<newnewhand::StellaStereoSlamTracker>(std::move(stella_config));
#endif
        } else {
            newnewhand::StereoVisualOdometryConfig slam_config;
            slam_config.calibration = calibration;
            slam_config.input_views_are_undistorted = false;
            slam_config.max_features = 5000;
            slam_config.num_levels = 10;
            slam_config.fast_threshold = 8;
            slam_config.temporal_ratio_test = 0.85f;
            slam_config.min_stereo_points = 25;
            slam_config.min_tracking_points = 15;
            slam_config.stereo_num_disparities = 256;
            slam_config.stereo_block_size = 7;
            slam_config.stereo_uniqueness_ratio = 5;
            slam_config.stereo_speckle_window_size = 100;
            slam_config.stereo_speckle_range = 4;
            slam_config.pnp_iterations = 200;
            slam_config.pnp_reprojection_error_px = 2.5f;
            slam_config.pnp_confidence = 0.999;
            slam_config.verbose_logging = options.verbose;
            legacy_slam_tracker = std::make_unique<newnewhand::StereoVisualOdometry>(std::move(slam_config));
        }

        std::unique_ptr<newnewhand::OfflineSequenceWriter> offline_writer;
        if (!options.offline_dump_dir.empty()) {
            newnewhand::OfflineSequenceWriterConfig writer_config;
            writer_config.output_root = options.offline_dump_dir;
            writer_config.calibration_source_path = calibration_path;
            writer_config.save_raw_images = true;
            writer_config.save_overlay_images = true;
            offline_writer = std::make_unique<newnewhand::OfflineSequenceWriter>(writer_config, calibration);
            offline_writer->Initialize();
        }

        const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(ParseDictionaryName(options.dictionary_name));
        cv::aruco::CharucoBoard board(
            cv::Size(options.squares_x, options.squares_y),
            options.square_length_m,
            options.marker_length_m,
            dictionary);
        board.setLegacyPattern(options.legacy_pattern);

        const cv::Mat undistorted_left_camera_matrix = BuildUndistortedCameraMatrix(
            calibration.left_camera_matrix,
            calibration.left_dist_coeffs,
            calibration.image_size);
        const cv::Mat undistorted_right_camera_matrix = BuildUndistortedCameraMatrix(
            calibration.right_camera_matrix,
            calibration.right_dist_coeffs,
            calibration.image_size);
        const cv::Mat undistorted_zero_dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);

        cv::aruco::CharucoParameters charuco_parameters;
        charuco_parameters.cameraMatrix = undistorted_left_camera_matrix;
        charuco_parameters.distCoeffs = undistorted_zero_dist_coeffs;
        cv::aruco::DetectorParameters detector_parameters;
        detector_parameters.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        cv::aruco::CharucoDetector detector(board, charuco_parameters, detector_parameters);

        cv::Mat left_map_x;
        cv::Mat left_map_y;
        cv::Mat right_map_x;
        cv::Mat right_map_y;
        cv::initUndistortRectifyMap(
            calibration.left_camera_matrix,
            calibration.left_dist_coeffs,
            cv::Mat(),
            undistorted_left_camera_matrix,
            calibration.image_size,
            CV_32FC1,
            left_map_x,
            left_map_y);
        cv::initUndistortRectifyMap(
            calibration.right_camera_matrix,
            calibration.right_dist_coeffs,
            cv::Mat(),
            undistorted_right_camera_matrix,
            calibration.image_size,
            CV_32FC1,
            right_map_x,
            right_map_y);

        newnewhand::GlfwSceneViewerConfig viewer_config;
        viewer_config.draw_world_axes = true;
        viewer_config.has_cam1_pose = true;
        viewer_config.cam1_rotation_cam1_to_cam0 = cv::Matx33f(calibration.rotation).t();
        const cv::Vec3f translation_01(calibration.translation);
        viewer_config.cam1_center_cam0 = -(viewer_config.cam1_rotation_cam1_to_cam0 * translation_01);
        newnewhand::GlfwSceneViewer viewer(viewer_config);
        if (options.glfw_view && !viewer.Initialize()) {
            throw std::runtime_error("failed to initialize GLFW OpenGL viewer");
        }

        const auto playback_interval = options.playback_fps == 0
            ? std::chrono::microseconds(0)
            : std::chrono::microseconds(1000000 / options.playback_fps);

        PreviousPoseState previous_pose_state;
        SlamAlignmentState slam_alignment_state;
        LastCalibratedPoseState last_calibrated_pose_state;
        HybridTrackingState hybrid_tracking_state;
        std::vector<newnewhand::OfflineCameraTrajectorySample> trajectory_samples;
        trajectory_samples.reserve(image_pairs.size());

        for (std::size_t pair_index = 0; pair_index < image_pairs.size(); ++pair_index) {
            const std::uint64_t capture_index =
                ParseCaptureIndexFromPath(image_pairs[pair_index].left_path, pair_index + 1);
            auto raw_stereo_frame = LoadStereoFrame(image_pairs[pair_index], capture_index, calibration);
            auto undistorted_stereo_frame = UndistortStereoFrame(
                raw_stereo_frame,
                left_map_x,
                left_map_y,
                right_map_x,
                right_map_y);

            newnewhand::StereoCameraTrackingResult slam_result;
            if (use_stella_backend) {
#ifdef NEWNEWHAND_HAVE_STELLA_VSLAM
                slam_result = stella_slam_tracker->Track(raw_stereo_frame);
#endif
            } else {
                slam_result = legacy_slam_tracker->Track(ToTrackingFrame(raw_stereo_frame));
            }
            BoardDetectionResult board_detection = EstimateBoardPoseInLeftCamera(
                undistorted_stereo_frame.views[0].bgr_image,
                board,
                detector,
                undistorted_left_camera_matrix,
                undistorted_zero_dist_coeffs,
                &previous_pose_state);
            auto initial_tracking_result = ResolveHybridTrackingResult(
                board_detection,
                slam_result,
                raw_stereo_frame.capture_index,
                raw_stereo_frame.trigger_timestamp,
                slam_alignment_state,
                last_calibrated_pose_state,
                hybrid_tracking_state);

            newnewhand::OfflineCameraTrajectorySample sample;
            sample.capture_index = raw_stereo_frame.capture_index;
            sample.trigger_timestamp = raw_stereo_frame.trigger_timestamp;
            sample.initial_tracking_result = initial_tracking_result;
            sample.slam_tracking_result = slam_result;
            sample.has_charuco_pose = board_detection.detected;
            sample.charuco_num_corners = board_detection.num_charuco_corners;
            sample.charuco_reprojection_error_px = board_detection.reprojection_error_px;
            sample.charuco_rotation_world_from_cam0 = board_detection.rotation_world_from_cam0;
            sample.charuco_camera_center_world = board_detection.camera_center_world;
            trajectory_samples.push_back(std::move(sample));
        }

        newnewhand::OfflineCameraTrajectoryOptimizerConfig optimizer_config;
        optimizer_config.verbose_logging = options.verbose;
        optimizer_config.slam_translation_sigma_m = 0.01;
        optimizer_config.slam_rotation_sigma_deg = 0.8;
        optimizer_config.charuco_translation_sigma_m = 0.002;
        optimizer_config.charuco_rotation_sigma_deg = 0.25;
        optimizer_config.min_charuco_corners = 6;
        optimizer_config.max_charuco_reprojection_error_px = 4.0f;
        newnewhand::OfflineCameraTrajectoryOptimizer optimizer(optimizer_config);
        const auto optimized_trajectory = optimizer.Optimize(trajectory_samples);

        double processing_fps = 0.0;
        auto previous_loop_time = std::chrono::steady_clock::now();
        PreviousPoseState preview_pose_state;

        for (std::size_t pair_index = 0; pair_index < image_pairs.size(); ++pair_index) {
            const auto loop_start = std::chrono::steady_clock::now();
            const std::uint64_t capture_index = trajectory_samples[pair_index].capture_index;
            auto raw_stereo_frame = LoadStereoFrame(image_pairs[pair_index], capture_index, calibration);
            auto undistorted_stereo_frame = UndistortStereoFrame(
                raw_stereo_frame,
                left_map_x,
                left_map_y,
                right_map_x,
                right_map_y);

            auto stereo_frame = pose_pipeline.Estimate(undistorted_stereo_frame);
            auto fused_frame = fuser.Fuse(stereo_frame);
            const auto& board_sample = trajectory_samples[pair_index];
            BoardDetectionResult board_detection = EstimateBoardPoseInLeftCamera(
                undistorted_stereo_frame.views[0].bgr_image,
                board,
                detector,
                undistorted_left_camera_matrix,
                undistorted_zero_dist_coeffs,
                &preview_pose_state);
            const auto& tracking_result = optimized_trajectory.optimized_tracking_results.at(pair_index);

            if (board_detection.detected && !stereo_frame.views[0].overlay_image.empty()) {
                cv::aruco::drawDetectedMarkers(
                    stereo_frame.views[0].overlay_image,
                    board_detection.marker_corners,
                    board_detection.marker_ids);
                cv::aruco::drawDetectedCornersCharuco(
                    stereo_frame.views[0].overlay_image,
                    board_detection.charuco_corners,
                    board_detection.charuco_ids);
                cv::drawFrameAxes(
                    stereo_frame.views[0].overlay_image,
                    undistorted_left_camera_matrix,
                    undistorted_zero_dist_coeffs,
                    board_detection.rvec,
                    board_detection.tvec,
                    options.square_length_m * 0.8f,
                    2);
            }

            const auto now = std::chrono::steady_clock::now();
            const double dt_seconds =
                std::chrono::duration<double>(now - previous_loop_time).count();
            previous_loop_time = now;
            if (dt_seconds > 1e-6) {
                const double instant_fps = 1.0 / dt_seconds;
                processing_fps = processing_fps <= 0.0
                    ? instant_fps
                    : 0.85 * processing_fps + 0.15 * instant_fps;
            }

            if (options.verbose) {
                std::cerr
                    << "[offline-fused] capture=" << tracking_result.capture_index
                    << " status=" << tracking_result.status_message
                    << " optimized_vertices=" << optimized_trajectory.num_vertices
                    << " optimized_slam_edges=" << optimized_trajectory.num_slam_edges
                    << " optimized_charuco_priors=" << optimized_trajectory.num_charuco_priors
                    << " board_detected=" << (board_sample.has_charuco_pose ? 1 : 0)
                    << " slam_initialized=" << (board_sample.slam_tracking_result.initialized ? 1 : 0)
                    << " slam_ok=" << (board_sample.slam_tracking_result.tracking_ok ? 1 : 0)
                    << " slam_reinit=" << (board_sample.slam_tracking_result.reinitialized ? 1 : 0)
                    << " hands=" << fused_frame.hands.size()
                    << " corners=" << board_sample.charuco_num_corners
                    << " slam_match=" << board_sample.slam_tracking_result.matched_points
                    << " slam_inlier=" << board_sample.slam_tracking_result.tracking_inliers
                    << " fps=" << processing_fps
                    << "\n";
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
                offline_writer->SaveFrame(raw_stereo_frame, stereo_frame, fused_frame);
            }

            if (options.glfw_view) {
                if (!viewer.Render(fused_frame, &tracking_result)) {
                    std::cerr << "[offline-fused] GLFW viewer closed by user\n";
                    break;
                }
            }

            if (options.preview) {
                const std::array<std::string, 2> labels = {"LEFT", "RIGHT"};
                for (std::size_t i = 0; i < stereo_frame.views.size(); ++i) {
                    if (stereo_frame.views[i].overlay_image.empty()) {
                        continue;
                    }
                    cv::Mat preview;
                    cv::resize(stereo_frame.views[i].overlay_image, preview, cv::Size(), 0.5, 0.5);
                    DrawTrackingOverlay(
                        preview,
                        labels[i],
                        board_detection,
                        tracking_result,
                        processing_fps);
                    cv::imshow("offline_fused_" + labels[i], preview);
                }

                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    break;
                }
            }

            if (playback_interval.count() > 0) {
                const auto elapsed = std::chrono::steady_clock::now() - loop_start;
                const auto remaining =
                    playback_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
                if (remaining.count() > 0) {
                    std::this_thread::sleep_for(remaining);
                }
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
