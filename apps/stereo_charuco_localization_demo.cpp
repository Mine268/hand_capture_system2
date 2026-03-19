#include <atomic>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/capture/stereo_capture.h"
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

std::string DefaultCalibrationPath() { return ProjectRoot() + "/resources/stereo_calibration.yaml"; }

struct DemoOptions {
    std::string calibration_path = DefaultCalibrationPath();
    std::string dictionary_name = "DICT_APRILTAG_36h11";
    int squares_x = 7;
    int squares_y = 5;
    float square_length_m = 0.04f;
    float marker_length_m = 0.03f;
    bool legacy_pattern = false;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 0;
    int frames = -1;
    bool preview = true;
    bool glfw_view = true;
    bool verbose = true;
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

struct LatestLocalizationData {
    newnewhand::StereoFrame stereo_frame;
    std::array<cv::Mat, 2> rectified_views;
    newnewhand::StereoCameraTrackingResult tracking_result;
    BoardDetectionResult left_detection;
    double localization_fps = 0.0;
};

struct SharedWorkerState {
    std::mutex mutex;
    std::shared_ptr<const LatestLocalizationData> latest_frame;
    std::exception_ptr worker_error;
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

        if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--dictionary") options.dictionary_name = require_value(arg);
        else if (arg == "--squares_x") options.squares_x = std::stoi(require_value(arg));
        else if (arg == "--squares_y") options.squares_y = std::stoi(require_value(arg));
        else if (arg == "--square_length_m") options.square_length_m = std::stof(require_value(arg));
        else if (arg == "--marker_length_m") options.marker_length_m = std::stof(require_value(arg));
        else if (arg == "--legacy_pattern") options.legacy_pattern = true;
        else if (arg == "--no_legacy_pattern") options.legacy_pattern = false;
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--fps") options.fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--glfw_view") options.glfw_view = true;
        else if (arg == "--no_glfw_view") options.glfw_view = false;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--quiet") options.verbose = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_charuco_localization_demo [options]\n"
                << "  --calibration <path>        default: " << DefaultCalibrationPath() << "\n"
                << "  --dictionary <name>         default: DICT_APRILTAG_36h11\n"
                << "  --squares_x <int>           default: 7\n"
                << "  --squares_y <int>           default: 5\n"
                << "  --square_length_m <float>   default: 0.04\n"
                << "  --marker_length_m <float>   default: 0.03\n"
                << "  --legacy_pattern            default: disabled\n"
                << "  --exposure_us <float>       default: 10000\n"
                << "  --gain <float>              default: -1 (auto)\n"
                << "  --fps <int>                 default: 0 (unlimited)\n"
                << "  --frames <int>              default: -1 (run until quit)\n"
                << "  --preview | --no_preview    default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
                << "  --verbose | --quiet         default: --verbose\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.squares_x < 3 || options.squares_y < 3) {
        throw std::runtime_error(
            "ChArUco pose estimation requires at least a 3x3 board; 2x2 is not sufficient");
    }
    if (options.square_length_m <= 0.0f || options.marker_length_m <= 0.0f) {
        throw std::runtime_error("--square_length_m and --marker_length_m must be positive");
    }
    if (options.marker_length_m >= options.square_length_m) {
        throw std::runtime_error("--marker_length_m must be smaller than --square_length_m");
    }
    return options;
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

newnewhand::StereoCaptureConfig BuildCaptureConfigFromCalibration(
    const DemoOptions& options,
    const newnewhand::StereoCalibrationResult& calibration) {
    if (calibration.left_camera_serial_number.empty() || calibration.right_camera_serial_number.empty()) {
        throw std::runtime_error(
            "calibration yaml is missing left_camera_serial_number/right_camera_serial_number");
    }
    newnewhand::StereoCaptureConfig capture_config;
    capture_config.serial_numbers = {
        calibration.left_camera_serial_number,
        calibration.right_camera_serial_number,
    };
    capture_config.camera_settings.exposure_us = options.exposure_us;
    capture_config.camera_settings.gain = options.gain;
    return capture_config;
}

void ValidateActiveCamerasAgainstCalibration(
    const std::array<newnewhand::CameraDescriptor, 2>& active_cameras,
    const newnewhand::StereoCalibrationResult& calibration) {
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

    const cv::Vec3f current_board_normal_in_cam = BoardNormalInCamera(rvec);
    if (previous_pose_state && previous_pose_state->initialized) {
        const float normal_cos =
            CosineSimilarity(current_board_normal_in_cam, previous_pose_state->board_normal_in_cam);
        if (normal_cos < -0.2f) {
            return result;
        }
    }

    result.rotation_world_from_cam0 = rotation_world_from_cam0;
    result.camera_center_world =
        -MatVecMul(rotation_world_from_cam0, translation_cam0_from_world);
    if (previous_pose_state) {
        previous_pose_state->initialized = true;
        previous_pose_state->rvec = rvec.clone();
        previous_pose_state->tvec = tvec.clone();
        previous_pose_state->board_normal_in_cam = current_board_normal_in_cam;
    }
    return result;
}

void DrawLocalizationOverlay(
    cv::Mat& image,
    const std::string& label,
    int num_charuco_corners,
    float reprojection_error_px,
    double localization_fps,
    double render_fps,
    const newnewhand::StereoCameraTrackingResult& tracking_result) {
    cv::putText(
        image,
        label,
        cv::Point(16, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(40, 220, 255),
        2);

    std::ostringstream status;
    status
        << "charuco corners=" << num_charuco_corners
        << " reproj=" << std::fixed << std::setprecision(3) << reprojection_error_px
        << " loc_fps=" << std::fixed << std::setprecision(1) << localization_fps
        << " render_fps=" << render_fps;
    cv::putText(
        image,
        status.str(),
        cv::Point(16, 56),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(40, 220, 255),
        1);

    if (!tracking_result.status_message.empty()) {
        cv::putText(
            image,
            tracking_result.status_message,
            cv::Point(16, 80),
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
            cv::Point(16, 104),
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
        const auto calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        const auto capture_config = BuildCaptureConfigFromCalibration(options, calibration);
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

        newnewhand::GlfwSceneViewerConfig viewer_config;
        viewer_config.draw_world_axes = true;
        viewer_config.draw_mesh = false;
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
        SharedWorkerState shared_state;
        std::atomic<bool> stop_requested = false;
        std::atomic<bool> worker_finished = false;
        std::thread worker([&]() {
            try {
                newnewhand::StereoCapture capture(capture_config);
                capture.Initialize();
                capture.Start();
                ValidateActiveCamerasAgainstCalibration(capture.ActiveCameras(), calibration);

                double localization_fps = 0.0;
                auto previous_loop_time = std::chrono::steady_clock::now();
                std::vector<cv::Vec3f> trajectory_world;
                bool has_pose = false;
                cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
                cv::Vec3f camera_center_world(0.0f, 0.0f, 0.0f);
                PreviousPoseState previous_pose_state;

                for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
                    if (stop_requested.load()) {
                        break;
                    }

                    const auto loop_start = std::chrono::steady_clock::now();
                    auto stereo_frame = capture.Capture();
                    if (stop_requested.load()) {
                        break;
                    }

                    std::array<cv::Mat, 2> rectified_views;
                    cv::remap(
                        stereo_frame.views[0].bgr_image,
                        rectified_views[0],
                        left_map_x,
                        left_map_y,
                        cv::INTER_LINEAR);
                    cv::remap(
                        stereo_frame.views[1].bgr_image,
                        rectified_views[1],
                        right_map_x,
                        right_map_y,
                        cv::INTER_LINEAR);

                    BoardDetectionResult left_detection = EstimateBoardPoseInLeftCamera(
                        rectified_views[0],
                        board,
                        detector,
                        undistorted_left_camera_matrix,
                        undistorted_zero_dist_coeffs,
                        &previous_pose_state);

                    newnewhand::StereoCameraTrackingResult localization_result;
                    localization_result.capture_index = stereo_frame.capture_index;
                    localization_result.trigger_timestamp = stereo_frame.trigger_timestamp;

                    if (left_detection.detected) {
                        has_pose = true;
                        rotation_world_from_cam0 = left_detection.rotation_world_from_cam0;
                        camera_center_world = left_detection.camera_center_world;
                    }

                    localization_result.status_message = left_detection.detected
                        ? "charuco board detected in left camera"
                        : (has_pose ? "charuco board lost, holding last pose" : "charuco board not detected");

                    if (has_pose) {
                        localization_result.initialized = true;
                        localization_result.tracking_ok = left_detection.detected;
                        localization_result.rotation_world_from_cam0 = rotation_world_from_cam0;
                        localization_result.camera_center_world = camera_center_world;
                        if (left_detection.detected) {
                            trajectory_world.push_back(camera_center_world);
                        }
                        localization_result.trajectory_world = trajectory_world;
                    }

                    const auto now = std::chrono::steady_clock::now();
                    const double dt_seconds =
                        std::chrono::duration<double>(now - previous_loop_time).count();
                    previous_loop_time = now;
                    if (dt_seconds > 1e-6) {
                        const double instant_fps = 1.0 / dt_seconds;
                        localization_fps = localization_fps <= 0.0
                            ? instant_fps
                            : 0.85 * localization_fps + 0.15 * instant_fps;
                    }

                    if (options.verbose) {
                        std::cerr
                            << "[charuco] capture=" << localization_result.capture_index
                            << " status=" << localization_result.status_message
                            << " loc_fps=" << localization_fps
                            << " corners=" << left_detection.num_charuco_corners
                            << " reproj=" << left_detection.reprojection_error_px
                            << "\n";
                    }

                    auto latest_frame = std::make_shared<LatestLocalizationData>();
                    latest_frame->stereo_frame = std::move(stereo_frame);
                    latest_frame->rectified_views = std::move(rectified_views);
                    latest_frame->tracking_result = std::move(localization_result);
                    latest_frame->left_detection = std::move(left_detection);
                    latest_frame->localization_fps = localization_fps;
                    {
                        std::lock_guard<std::mutex> lock(shared_state.mutex);
                        shared_state.latest_frame = std::move(latest_frame);
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

                capture.Stop();
            } catch (...) {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                shared_state.worker_error = std::current_exception();
            }
            worker_finished.store(true);
        });

        std::shared_ptr<const LatestLocalizationData> latest_frame;
        std::exception_ptr worker_error;
        const newnewhand::StereoFusedHandPoseFrame empty_fused_frame;
        double render_fps = 0.0;
        auto previous_render_time = std::chrono::steady_clock::now();

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

            const auto render_now = std::chrono::steady_clock::now();
            const double render_dt_seconds =
                std::chrono::duration<double>(render_now - previous_render_time).count();
            previous_render_time = render_now;
            if (render_dt_seconds > 1e-6) {
                const double instant_render_fps = 1.0 / render_dt_seconds;
                render_fps = render_fps <= 0.0
                    ? instant_render_fps
                    : 0.85 * render_fps + 0.15 * instant_render_fps;
            }

            if (options.glfw_view) {
                const newnewhand::StereoCameraTrackingResult* tracking_result =
                    latest_frame ? &latest_frame->tracking_result : nullptr;
                std::ostringstream title;
                title
                    << "newnewhand ChArUco Localization"
                    << " | loc "
                    << std::fixed << std::setprecision(1)
                    << (latest_frame ? latest_frame->localization_fps : 0.0)
                    << " FPS"
                    << " | render "
                    << render_fps
                    << " FPS";
                viewer.SetTitle(title.str());
                if (!viewer.Render(empty_fused_frame, tracking_result)) {
                    std::cerr << "[charuco] GLFW viewer closed by user\n";
                    stop_requested.store(true);
                    break;
                }
            }

            if (options.preview) {
                if (latest_frame) {
                    std::array<cv::Mat, 2> previews = {
                        latest_frame->rectified_views[0].empty()
                            ? latest_frame->stereo_frame.views[0].bgr_image.clone()
                            : latest_frame->rectified_views[0].clone(),
                        latest_frame->rectified_views[1].empty()
                            ? latest_frame->stereo_frame.views[1].bgr_image.clone()
                            : latest_frame->rectified_views[1].clone(),
                    };
                    const std::array<std::string, 2> labels = {"LEFT", "RIGHT"};

                    if (latest_frame->left_detection.detected) {
                        cv::aruco::drawDetectedMarkers(
                            previews[0],
                            latest_frame->left_detection.marker_corners,
                            latest_frame->left_detection.marker_ids);
                        cv::aruco::drawDetectedCornersCharuco(
                            previews[0],
                            latest_frame->left_detection.charuco_corners,
                            latest_frame->left_detection.charuco_ids);
                        cv::drawFrameAxes(
                            previews[0],
                            undistorted_left_camera_matrix,
                            undistorted_zero_dist_coeffs,
                            latest_frame->left_detection.rvec,
                            latest_frame->left_detection.tvec,
                            options.square_length_m * 0.8f,
                            2);
                    }

                    for (std::size_t i = 0; i < previews.size(); ++i) {
                        if (previews[i].empty()) {
                            continue;
                        }
                        DrawLocalizationOverlay(
                            previews[i],
                            labels[i],
                            latest_frame->left_detection.num_charuco_corners,
                            latest_frame->left_detection.reprojection_error_px,
                            latest_frame->localization_fps,
                            render_fps,
                            latest_frame->tracking_result);
                        cv::resize(previews[i], previews[i], cv::Size(), 0.5, 0.5);
                        cv::imshow("charuco_" + labels[i], previews[i]);
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
