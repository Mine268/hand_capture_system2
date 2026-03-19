#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
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
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/video/tracking.hpp>

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
    int marker_id = 0;
    float marker_length_m = 0.12f;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 0;
    int frames = -1;
    bool preview = true;
    bool glfw_view = true;
    bool verbose = true;
};

struct MarkerDetectionResult {
    bool detected = false;
    bool used_right_camera_fallback = false;
    int marker_id = -1;
    float reprojection_error_px = 0.0f;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Matx33f rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
    std::vector<cv::Point2f> corners;
};

struct LatestLocalizationData {
    newnewhand::StereoFrame stereo_frame;
    newnewhand::StereoCameraTrackingResult tracking_result;
    MarkerDetectionResult left_detection;
    double localization_fps = 0.0;
};

struct SharedWorkerState {
    std::mutex mutex;
    std::shared_ptr<const LatestLocalizationData> latest_frame;
    std::exception_ptr worker_error;
};

struct Quaternion {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct PositionKalmanFilter {
    bool initialized = false;
    std::chrono::steady_clock::time_point last_timestamp;
    cv::KalmanFilter filter;

    PositionKalmanFilter() {
        filter.init(6, 3, 0, CV_32F);
        filter.measurementMatrix = (cv::Mat_<float>(3, 6) <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0);
        cv::setIdentity(filter.processNoiseCov, cv::Scalar::all(1e-4f));
        cv::setIdentity(filter.measurementNoiseCov, cv::Scalar::all(2e-4f));
        cv::setIdentity(filter.errorCovPost, cv::Scalar::all(1e-2f));
        cv::setIdentity(filter.errorCovPre, cv::Scalar::all(1e-2f));
    }
};

struct QuaternionKalmanFilter {
    bool initialized = false;
    cv::KalmanFilter filter;

    QuaternionKalmanFilter() {
        filter.init(4, 4, 0, CV_32F);
        cv::setIdentity(filter.transitionMatrix);
        cv::setIdentity(filter.measurementMatrix);
        cv::setIdentity(filter.processNoiseCov, cv::Scalar::all(8e-5f));
        cv::setIdentity(filter.measurementNoiseCov, cv::Scalar::all(3e-4f));
        cv::setIdentity(filter.errorCovPost, cv::Scalar::all(1e-2f));
        cv::setIdentity(filter.errorCovPre, cv::Scalar::all(1e-2f));
    }
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
        else if (arg == "--marker_id") options.marker_id = std::stoi(require_value(arg));
        else if (arg == "--marker_length_m") options.marker_length_m = std::stof(require_value(arg));
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
                << "Usage: stereo_aruco_localization_demo [options]\n"
                << "  --calibration <path>      default: " << DefaultCalibrationPath() << "\n"
                << "  --dictionary <name>       default: DICT_APRILTAG_36h11\n"
                << "  --marker_id <int>         default: 0\n"
                << "  --marker_length_m <float> default: 0.12\n"
                << "  --exposure_us <float>     default: 10000\n"
                << "  --gain <float>            default: -1 (auto)\n"
                << "  --fps <int>               default: 0 (unlimited)\n"
                << "  --frames <int>            default: -1 (run until quit)\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
                << "  --verbose | --quiet       default: --verbose\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.marker_length_m <= 0.0f) {
        throw std::runtime_error("--marker_length_m must be positive");
    }
    return options;
}

int ParseDictionaryName(const std::string& name) {
    static const std::unordered_map<std::string, int> kDictionaryMap = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
    };

    const auto it = kDictionaryMap.find(name);
    if (it == kDictionaryMap.end()) {
        throw std::runtime_error("unsupported ArUco/AprilTag dictionary: " + name);
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

std::vector<cv::Point3f> MarkerObjectPoints(float marker_length_m) {
    const float half = marker_length_m * 0.5f;
    return {
        {-half, half, 0.0f},
        {half, half, 0.0f},
        {half, -half, 0.0f},
        {-half, -half, 0.0f},
    };
}

cv::Matx33f ToMatx33f(const cv::Mat& matrix) {
    cv::Mat matrix32f;
    matrix.convertTo(matrix32f, CV_32F);
    cv::Matx33f output = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            output(r, c) = matrix32f.at<float>(r, c);
        }
    }
    return output;
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

float Dot(const Quaternion& a, const Quaternion& b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

Quaternion Normalize(const Quaternion& q) {
    const float norm = std::sqrt(Dot(q, q));
    if (norm <= 1e-8f) {
        return {};
    }
    return {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

Quaternion Negate(const Quaternion& q) {
    return {-q.w, -q.x, -q.y, -q.z};
}

Quaternion AlignQuaternionSign(Quaternion q, const Quaternion& reference) {
    if (Dot(q, reference) < 0.0f) {
        q = Negate(q);
    }
    return q;
}

Quaternion QuaternionFromRotationMatrix(const cv::Matx33f& r) {
    Quaternion q;
    const float trace = r(0, 0) + r(1, 1) + r(2, 2);
    if (trace > 0.0f) {
        const float s = 2.0f * std::sqrt(trace + 1.0f);
        q.w = 0.25f * s;
        q.x = (r(2, 1) - r(1, 2)) / s;
        q.y = (r(0, 2) - r(2, 0)) / s;
        q.z = (r(1, 0) - r(0, 1)) / s;
    } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
        const float s = 2.0f * std::sqrt(1.0f + r(0, 0) - r(1, 1) - r(2, 2));
        q.w = (r(2, 1) - r(1, 2)) / s;
        q.x = 0.25f * s;
        q.y = (r(0, 1) + r(1, 0)) / s;
        q.z = (r(0, 2) + r(2, 0)) / s;
    } else if (r(1, 1) > r(2, 2)) {
        const float s = 2.0f * std::sqrt(1.0f + r(1, 1) - r(0, 0) - r(2, 2));
        q.w = (r(0, 2) - r(2, 0)) / s;
        q.x = (r(0, 1) + r(1, 0)) / s;
        q.y = 0.25f * s;
        q.z = (r(1, 2) + r(2, 1)) / s;
    } else {
        const float s = 2.0f * std::sqrt(1.0f + r(2, 2) - r(0, 0) - r(1, 1));
        q.w = (r(1, 0) - r(0, 1)) / s;
        q.x = (r(0, 2) + r(2, 0)) / s;
        q.y = (r(1, 2) + r(2, 1)) / s;
        q.z = 0.25f * s;
    }
    return Normalize(q);
}

cv::Matx33f RotationMatrixFromQuaternion(const Quaternion& q_in) {
    const Quaternion q = Normalize(q_in);
    const float ww = q.w * q.w;
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    return cv::Matx33f(
        ww + xx - yy - zz, 2.0f * (xy - wz), 2.0f * (xz + wy),
        2.0f * (xy + wz), ww - xx + yy - zz, 2.0f * (yz - wx),
        2.0f * (xz - wy), 2.0f * (yz + wx), ww - xx - yy + zz);
}

cv::Mat QuaternionToColumnMat(const Quaternion& q) {
    return (cv::Mat_<float>(4, 1) << q.w, q.x, q.y, q.z);
}

Quaternion QuaternionFromColumnMat(const cv::Mat& q_mat) {
    cv::Mat q32f;
    q_mat.convertTo(q32f, CV_32F);
    return Normalize({
        q32f.at<float>(0, 0),
        q32f.at<float>(1, 0),
        q32f.at<float>(2, 0),
        q32f.at<float>(3, 0),
    });
}

float MeanCornerReprojectionError(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected);
    if (projected.size() != image_points.size() || projected.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    float error = 0.0f;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        error += cv::norm(projected[i] - image_points[i]);
    }
    return error / static_cast<float>(projected.size());
}

void RefineMarkerCorners(const cv::Mat& bgr_image, std::vector<cv::Point2f>& corners) {
    if (corners.empty()) {
        return;
    }
    cv::Mat gray;
    cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(
        gray,
        corners,
        cv::Size(5, 5),
        cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
}

void UpdatePositionTransition(PositionKalmanFilter& filter_state, float dt_seconds) {
    const float dt = std::clamp(dt_seconds, 1e-3f, 0.2f);
    filter_state.filter.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
}

cv::Vec3f UpdatePositionFilter(
    PositionKalmanFilter& filter_state,
    const cv::Vec3f* measured_position,
    std::chrono::steady_clock::time_point timestamp) {
    if (!filter_state.initialized) {
        if (!measured_position) {
            return cv::Vec3f(0.0f, 0.0f, 0.0f);
        }
        UpdatePositionTransition(filter_state, 0.1f);
        filter_state.filter.statePost = (cv::Mat_<float>(6, 1) <<
            (*measured_position)[0], (*measured_position)[1], (*measured_position)[2],
            0.0f, 0.0f, 0.0f);
        filter_state.filter.statePre = filter_state.filter.statePost.clone();
        filter_state.last_timestamp = timestamp;
        filter_state.initialized = true;
        return *measured_position;
    }

    const float dt_seconds =
        std::chrono::duration<float>(timestamp - filter_state.last_timestamp).count();
    filter_state.last_timestamp = timestamp;
    UpdatePositionTransition(filter_state, dt_seconds);
    cv::Mat predicted = filter_state.filter.predict();
    if (!measured_position) {
        return {
            predicted.at<float>(0, 0),
            predicted.at<float>(1, 0),
            predicted.at<float>(2, 0),
        };
    }

    const cv::Mat measurement =
        (cv::Mat_<float>(3, 1) << (*measured_position)[0], (*measured_position)[1], (*measured_position)[2]);
    const cv::Mat corrected = filter_state.filter.correct(measurement);
    return {
        corrected.at<float>(0, 0),
        corrected.at<float>(1, 0),
        corrected.at<float>(2, 0),
    };
}

Quaternion UpdateOrientationFilter(
    QuaternionKalmanFilter& filter_state,
    const Quaternion* measured_orientation) {
    if (!filter_state.initialized) {
        if (!measured_orientation) {
            return {};
        }
        const Quaternion initial = Normalize(*measured_orientation);
        filter_state.filter.statePost = QuaternionToColumnMat(initial);
        filter_state.filter.statePre = filter_state.filter.statePost.clone();
        filter_state.initialized = true;
        return initial;
    }

    Quaternion predicted = QuaternionFromColumnMat(filter_state.filter.predict());
    filter_state.filter.statePre = QuaternionToColumnMat(predicted);
    if (!measured_orientation) {
        filter_state.filter.statePost = filter_state.filter.statePre.clone();
        return predicted;
    }

    const Quaternion aligned_measurement =
        AlignQuaternionSign(Normalize(*measured_orientation), predicted);
    const cv::Mat corrected = filter_state.filter.correct(QuaternionToColumnMat(aligned_measurement));
    const Quaternion normalized_corrected = QuaternionFromColumnMat(corrected);
    filter_state.filter.statePost = QuaternionToColumnMat(normalized_corrected);
    filter_state.filter.statePre = filter_state.filter.statePost.clone();
    return normalized_corrected;
}

MarkerDetectionResult FuseMarkerDetections(
    const MarkerDetectionResult& left_detection,
    const MarkerDetectionResult& right_detection_left_frame) {
    if (!left_detection.detected) {
        return right_detection_left_frame;
    }
    if (!right_detection_left_frame.detected) {
        return left_detection;
    }

    const float left_weight = 1.0f / std::max(1e-3f, left_detection.reprojection_error_px);
    const float right_weight = 1.0f / std::max(1e-3f, right_detection_left_frame.reprojection_error_px);
    const float weight_sum = left_weight + right_weight;

    MarkerDetectionResult fused = left_detection;
    fused.used_right_camera_fallback = false;
    fused.reprojection_error_px =
        (left_detection.reprojection_error_px * left_weight
         + right_detection_left_frame.reprojection_error_px * right_weight) / weight_sum;
    fused.camera_center_world =
        (left_detection.camera_center_world * left_weight
         + right_detection_left_frame.camera_center_world * right_weight) / weight_sum;

    Quaternion left_q = QuaternionFromRotationMatrix(left_detection.rotation_world_from_cam0);
    Quaternion right_q = AlignQuaternionSign(
        QuaternionFromRotationMatrix(right_detection_left_frame.rotation_world_from_cam0),
        left_q);
    Quaternion fused_q = Normalize({
        left_q.w * left_weight + right_q.w * right_weight,
        left_q.x * left_weight + right_q.x * right_weight,
        left_q.y * left_weight + right_q.y * right_weight,
        left_q.z * left_weight + right_q.z * right_weight,
    });
    fused.rotation_world_from_cam0 = RotationMatrixFromQuaternion(fused_q);
    return fused;
}

MarkerDetectionResult EstimateMarkerPoseInCamera(
    const cv::Mat& bgr_image,
    int target_marker_id,
    float marker_length_m,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::aruco::ArucoDetector& detector) {
    MarkerDetectionResult result;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;
    detector.detectMarkers(bgr_image, corners, ids, rejected);

    if (ids.empty()) {
        return result;
    }

    for (std::size_t marker_index = 0; marker_index < ids.size(); ++marker_index) {
        if (ids[marker_index] != target_marker_id) {
            continue;
        }

        std::vector<cv::Point2f> refined_corners = corners[marker_index];
        RefineMarkerCorners(bgr_image, refined_corners);

        cv::Mat rvec;
        cv::Mat tvec;
        const auto object_points = MarkerObjectPoints(marker_length_m);
        const bool pnp_ok = cv::solvePnP(
            object_points,
            refined_corners,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE);
        if (!pnp_ok) {
            continue;
        }

        cv::solvePnPRefineLM(
            object_points,
            refined_corners,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec);

        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        const cv::Matx33f rotation_camera_from_world = ToMatx33f(rotation_matrix);
        const cv::Matx33f rotation_world_from_camera = rotation_camera_from_world.t();
        const cv::Vec3f translation_camera_from_world = ToVec3f(tvec);

        result.detected = true;
        result.marker_id = ids[marker_index];
        result.corners = refined_corners;
        result.rvec = rvec;
        result.tvec = tvec;
        result.reprojection_error_px = MeanCornerReprojectionError(
            object_points,
            refined_corners,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs);
        result.rotation_world_from_cam0 = rotation_world_from_camera;
        result.camera_center_world =
            -MatVecMul(rotation_world_from_camera, translation_camera_from_world);
        return result;
    }

    return result;
}

MarkerDetectionResult ConvertRightMarkerPoseToLeftCameraPose(
    const MarkerDetectionResult& right_detection,
    const newnewhand::StereoCalibrationResult& calibration) {
    MarkerDetectionResult left_pose = right_detection;
    left_pose.used_right_camera_fallback = true;

    const cv::Matx33f rotation_right_from_left = ToMatx33f(calibration.rotation);
    cv::Mat translation32f;
    calibration.translation.convertTo(translation32f, CV_32F);
    const cv::Vec3f translation_right_from_left(
        translation32f.at<float>(0, 0),
        translation32f.at<float>(1, 0),
        translation32f.at<float>(2, 0));

    const cv::Matx33f rotation_world_from_right = right_detection.rotation_world_from_cam0;
    const cv::Matx33f rotation_right_from_world = rotation_world_from_right.t();
    const cv::Vec3f translation_right_from_world =
        -MatVecMul(rotation_right_from_world, right_detection.camera_center_world);

    const cv::Matx33f rotation_left_from_world = rotation_right_from_left.t() * rotation_right_from_world;
    const cv::Vec3f translation_left_from_world =
        MatVecMul(rotation_right_from_left.t(), translation_right_from_world - translation_right_from_left);

    left_pose.rotation_world_from_cam0 = rotation_left_from_world.t();
    left_pose.camera_center_world =
        -MatVecMul(left_pose.rotation_world_from_cam0, translation_left_from_world);
    return left_pose;
}

void DrawLocalizationOverlay(
    cv::Mat& image,
    const std::string& label,
    int marker_id,
    bool detected,
    bool used_right_fallback,
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
        << "marker " << marker_id
        << " "
        << (detected ? (used_right_fallback ? "right-fallback" : "detected") : "missing")
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
        const auto dictionary =
            cv::aruco::getPredefinedDictionary(ParseDictionaryName(options.dictionary_name));
        const cv::aruco::DetectorParameters detector_parameters;
        const cv::aruco::ArucoDetector detector(dictionary, detector_parameters);

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

        SharedWorkerState shared_state;
        std::atomic<bool> stop_requested = false;
        std::atomic<bool> worker_finished = false;
        std::thread worker([&]() {
            try {
                newnewhand::StereoCapture capture(capture_config);
                capture.Initialize();
                capture.Start();
                const auto active_cameras = capture.ActiveCameras();
                ValidateActiveCamerasAgainstCalibration(active_cameras, calibration);

                double localization_fps = 0.0;
                auto previous_loop_time = std::chrono::steady_clock::now();
                std::vector<cv::Vec3f> trajectory_world;
                PositionKalmanFilter position_filter;
                QuaternionKalmanFilter orientation_filter;

                try {
                    for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
                        if (stop_requested.load()) {
                            break;
                        }

                        const auto loop_start = std::chrono::steady_clock::now();
                        auto stereo_frame = capture.Capture();
                        if (stop_requested.load()) {
                            break;
                        }

                        MarkerDetectionResult left_detection = EstimateMarkerPoseInCamera(
                            stereo_frame.views[0].bgr_image,
                            options.marker_id,
                            options.marker_length_m,
                            calibration.left_camera_matrix,
                            calibration.left_dist_coeffs,
                            detector);
                        newnewhand::StereoCameraTrackingResult localization_result;
                        localization_result.capture_index = stereo_frame.capture_index;
                        localization_result.trigger_timestamp = stereo_frame.trigger_timestamp;

                        const cv::Vec3f* measured_position =
                            left_detection.detected ? &left_detection.camera_center_world : nullptr;
                        const Quaternion measured_orientation =
                            left_detection.detected
                            ? QuaternionFromRotationMatrix(left_detection.rotation_world_from_cam0)
                            : Quaternion{};
                        const Quaternion* measured_orientation_ptr =
                            left_detection.detected ? &measured_orientation : nullptr;

                        const cv::Vec3f filtered_position =
                            UpdatePositionFilter(position_filter, measured_position, stereo_frame.trigger_timestamp);
                        const Quaternion filtered_orientation =
                            UpdateOrientationFilter(orientation_filter, measured_orientation_ptr);

                        if (left_detection.detected) {
                            localization_result.status_message =
                                "marker detected in left camera";
                        } else {
                            localization_result.status_message =
                                position_filter.initialized
                                ? "marker lost, holding filtered pose"
                                : "marker not detected";
                        }

                        if (position_filter.initialized && orientation_filter.initialized) {
                            localization_result.initialized = true;
                            localization_result.tracking_ok = left_detection.detected;
                            localization_result.rotation_world_from_cam0 =
                                RotationMatrixFromQuaternion(filtered_orientation);
                            localization_result.camera_center_world = filtered_position;
                            if (left_detection.detected) {
                                trajectory_world.push_back(filtered_position);
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
                                << "[aruco] capture=" << localization_result.capture_index
                                << " status=" << localization_result.status_message
                                << " localization_fps=" << localization_fps
                                << " left_detected=" << left_detection.detected
                                << " left_err=" << left_detection.reprojection_error_px
                                << "\n";
                        }

                        auto latest_frame = std::make_shared<LatestLocalizationData>();
                        latest_frame->stereo_frame = std::move(stereo_frame);
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
                } catch (...) {
                    capture.Stop();
                    throw;
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
                    << "newnewhand ArUco Localization"
                    << " | loc "
                    << std::fixed << std::setprecision(1)
                    << (latest_frame ? latest_frame->localization_fps : 0.0)
                    << " FPS"
                    << " | render "
                    << render_fps
                    << " FPS";
                viewer.SetTitle(title.str());
                if (!viewer.Render(empty_fused_frame, tracking_result)) {
                    std::cerr << "[aruco] GLFW viewer closed by user\n";
                    stop_requested.store(true);
                    break;
                }
            }

            if (options.preview) {
                if (latest_frame) {
                    std::array<cv::Mat, 2> previews = {
                        latest_frame->stereo_frame.views[0].bgr_image.clone(),
                        latest_frame->stereo_frame.views[1].bgr_image.clone(),
                    };
                    std::array<MarkerDetectionResult, 2> detections = {
                        latest_frame->left_detection,
                        MarkerDetectionResult{},
                    };
                    std::array<cv::Mat, 2> camera_matrices = {
                        calibration.left_camera_matrix,
                        calibration.right_camera_matrix,
                    };
                    std::array<cv::Mat, 2> dist_coeffs = {
                        calibration.left_dist_coeffs,
                        calibration.right_dist_coeffs,
                    };
                    const std::array<std::string, 2> labels = {"LEFT", "RIGHT"};

                    for (std::size_t i = 0; i < previews.size(); ++i) {
                        if (previews[i].empty()) {
                            continue;
                        }
                        if (detections[i].detected) {
                            std::vector<std::vector<cv::Point2f>> marker_corners = {detections[i].corners};
                            std::vector<int> marker_ids = {detections[i].marker_id};
                            cv::aruco::drawDetectedMarkers(previews[i], marker_corners, marker_ids);
                            cv::drawFrameAxes(
                                previews[i],
                                camera_matrices[i],
                                dist_coeffs[i],
                                detections[i].rvec,
                                detections[i].tvec,
                                options.marker_length_m * 0.6f,
                                2);
                        }
                        DrawLocalizationOverlay(
                            previews[i],
                            labels[i],
                            options.marker_id,
                            detections[i].detected,
                            false,
                            latest_frame->localization_fps,
                            render_fps,
                            latest_frame->tracking_result);
                        cv::resize(previews[i], previews[i], cv::Size(), 0.5, 0.5);
                        cv::imshow("aruco_" + labels[i], previews[i]);
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
