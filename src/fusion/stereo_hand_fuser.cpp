#include "newnewhand/fusion/stereo_hand_fuser.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>

#include "hand_pose_utils.h"
#include "newnewhand/visualization/hand_pose_overlay.h"

namespace newnewhand {

namespace {

const HandPoseResult* FindByHandedness(const std::vector<HandPoseResult>& hands, bool is_right) {
    for (const auto& hand : hands) {
        if (hand.detection.is_right == is_right) {
            return &hand;
        }
    }
    return nullptr;
}

cv::Rect2f ComputeBBoxFromPoints(const float keypoints_2d[21][2]) {
    float min_x = keypoints_2d[0][0];
    float min_y = keypoints_2d[0][1];
    float max_x = keypoints_2d[0][0];
    float max_y = keypoints_2d[0][1];
    for (int i = 1; i < 21; ++i) {
        min_x = std::min(min_x, keypoints_2d[i][0]);
        min_y = std::min(min_y, keypoints_2d[i][1]);
        max_x = std::max(max_x, keypoints_2d[i][0]);
        max_y = std::max(max_y, keypoints_2d[i][1]);
    }
    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

std::string FormatPoint(const cv::Point2f& point) {
    std::ostringstream oss;
    oss << "(" << point.x << ", " << point.y << ")";
    return oss.str();
}

float ClampDtSeconds(float dt_seconds, float min_dt_seconds, float max_dt_seconds) {
    return std::clamp(dt_seconds, min_dt_seconds, max_dt_seconds);
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

cv::Mat ZeroDistCoeffsLike(const cv::Mat& dist_coeffs) {
    if (dist_coeffs.empty()) {
        return cv::Mat::zeros(1, 5, CV_64F);
    }
    return cv::Mat::zeros(dist_coeffs.rows, dist_coeffs.cols, dist_coeffs.type());
}

}  // namespace

StereoHandFuser::StereoHandFuser(StereoHandFuserConfig config)
    : config_(std::move(config)) {
    if (!config_.calibration.success) {
        throw std::invalid_argument("stereo calibration must be loaded before fusion");
    }

    if (config_.input_views_are_undistorted) {
        view0_camera_matrix_ = BuildUndistortedCameraMatrix(
            config_.calibration.left_camera_matrix,
            config_.calibration.left_dist_coeffs,
            config_.calibration.image_size);
        view1_camera_matrix_ = BuildUndistortedCameraMatrix(
            config_.calibration.right_camera_matrix,
            config_.calibration.right_dist_coeffs,
            config_.calibration.image_size);
        view0_dist_coeffs_ = ZeroDistCoeffsLike(config_.calibration.left_dist_coeffs);
        view1_dist_coeffs_ = ZeroDistCoeffsLike(config_.calibration.right_dist_coeffs);
    } else {
        view0_camera_matrix_ = config_.calibration.left_camera_matrix.clone();
        view1_camera_matrix_ = config_.calibration.right_camera_matrix.clone();
        view0_dist_coeffs_ = config_.calibration.left_dist_coeffs.clone();
        view1_dist_coeffs_ = config_.calibration.right_dist_coeffs.clone();
    }

    for (auto& state : root_filters_) {
        state.filter.init(6, 3, 0, CV_32F);
        state.filter.measurementMatrix = (cv::Mat_<float>(3, 6) <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0);
        cv::setIdentity(state.filter.measurementNoiseCov, cv::Scalar::all(config_.root_measurement_noise));
        cv::setIdentity(state.filter.errorCovPost, cv::Scalar::all(config_.root_initial_error));
        cv::setIdentity(state.filter.errorCovPre, cv::Scalar::all(config_.root_initial_error));
        UpdateFilterModel(state, config_.root_max_dt_seconds);
    }
}

StereoFusedHandPoseFrame StereoHandFuser::Fuse(const StereoSingleViewPoseFrame& stereo_frame) {
    StereoFusedHandPoseFrame fused_frame;
    fused_frame.capture_index = stereo_frame.capture_index;
    fused_frame.trigger_timestamp = stereo_frame.trigger_timestamp;

    if (config_.verbose_logging) {
        std::cerr
            << "[fusion] capture=" << stereo_frame.capture_index
            << " view0_hands=" << stereo_frame.views[0].hand_poses.size()
            << " view1_hands=" << stereo_frame.views[1].hand_poses.size()
            << "\n";
    }

    for (bool is_right : {false, true}) {
        FusedHandPose fused_hand = FuseHandByHandedness(stereo_frame, is_right);
        if (!fused_hand.fused_from_stereo) {
            continue;
        }
        fused_frame.hands.push_back(std::move(fused_hand));
    }

    return fused_frame;
}

void StereoHandFuser::SaveFrame(const StereoFusedHandPoseFrame& frame, const std::filesystem::path& output_path) const {
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    cv::FileStorage fs(output_path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open fused hand pose output file: " + output_path.string());
    }

    fs << "capture_index" << static_cast<double>(frame.capture_index);
    fs << "num_hands" << static_cast<int>(frame.hands.size());
    fs << "hands" << "[";
    for (const auto& hand : frame.hands) {
        fs << "{";
        fs << "is_right" << static_cast<int>(hand.is_right);
        fs << "fused_from_stereo" << static_cast<int>(hand.fused_from_stereo);
        fs << "has_view0" << static_cast<int>(hand.has_view0);
        fs << "has_view1" << static_cast<int>(hand.has_view1);
        fs << "root_joint_cam0" << cv::Mat(hand.root_joint_cam0);
        fs << "pred_cam" << cv::Mat(3, 1, CV_32F, const_cast<float*>(hand.pose_cam0.pred_cam)).clone();
        fs << "camera_translation" << cv::Mat(3, 1, CV_32F, const_cast<float*>(hand.pose_cam0.camera_translation)).clone();
        fs << "global_orient" << cv::Mat(3, 1, CV_32F, const_cast<float*>(hand.pose_cam0.global_orient)).clone();
        fs << "hand_pose" << cv::Mat(15, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.hand_pose[0][0])).clone();
        fs << "betas" << cv::Mat(10, 1, CV_32F, const_cast<float*>(hand.pose_cam0.betas)).clone();
        fs << "keypoints_2d" << cv::Mat(21, 2, CV_32F, const_cast<float*>(&hand.pose_cam0.keypoints_2d[0][0])).clone();
        fs << "keypoints_3d_cam0" << cv::Mat(21, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.keypoints_3d[0][0])).clone();
        fs << "vertices_cam0" << cv::Mat(778, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.vertices[0][0])).clone();
        fs << "}";
    }
    fs << "]";
}

FusedHandPose StereoHandFuser::FuseHandByHandedness(const StereoSingleViewPoseFrame& frame, bool is_right) {
    const HandPoseResult* view0 = FindByHandedness(frame.views[0].hand_poses, is_right);
    const HandPoseResult* view1 = FindByHandedness(frame.views[1].hand_poses, is_right);

    FusedHandPose fused;
    fused.is_right = is_right;
    fused.has_view0 = view0 != nullptr;
    fused.has_view1 = view1 != nullptr;

    if (config_.verbose_logging) {
        std::cerr
            << "[fusion] handedness=" << (is_right ? "right" : "left")
            << " has_view0=" << fused.has_view0
            << " has_view1=" << fused.has_view1;
        if (view0) {
            std::cerr << " root0=" << FormatPoint(cv::Point2f(view0->keypoints_2d[0][0], view0->keypoints_2d[0][1]));
        }
        if (view1) {
            std::cerr << " root1=" << FormatPoint(cv::Point2f(view1->keypoints_2d[0][0], view1->keypoints_2d[0][1]));
        }
        std::cerr << "\n";
    }

    if (!view0 || !view1) {
        MarkMissing(is_right, frame.trigger_timestamp);
        if (config_.verbose_logging && config_.require_both_views) {
            std::cerr
                << "[fusion] skip handedness=" << (is_right ? "right" : "left")
                << " reason=missing_view\n";
        }
        return fused;
    }

    fused.pose_cam0 = *view0;
    const cv::Vec3f triangulated_root = TriangulateRoot(
        cv::Point2f(view0->keypoints_2d[0][0], view0->keypoints_2d[0][1]),
        cv::Point2f(view1->keypoints_2d[0][0], view1->keypoints_2d[0][1]));
    fused.root_joint_cam0 = FilterRoot(triangulated_root, is_right, frame.trigger_timestamp);
    fused.pose_cam0.camera_translation[0] = fused.root_joint_cam0[0] - fused.pose_cam0.keypoints_3d[0][0];
    fused.pose_cam0.camera_translation[1] = fused.root_joint_cam0[1] - fused.pose_cam0.keypoints_3d[0][1];
    fused.pose_cam0.camera_translation[2] = fused.root_joint_cam0[2] - fused.pose_cam0.keypoints_3d[0][2];
    ProjectToView0(fused.pose_cam0);
    fused.fused_from_stereo = true;

    if (config_.verbose_logging) {
        std::cerr
            << "[fusion] handedness=" << (is_right ? "right" : "left")
            << " triangulated_root=("
            << triangulated_root[0] << ", "
            << triangulated_root[1] << ", "
            << triangulated_root[2] << ") "
            << "filtered_root=("
            << fused.root_joint_cam0[0] << ", "
            << fused.root_joint_cam0[1] << ", "
            << fused.root_joint_cam0[2] << ") "
            << "cam_t=("
            << fused.pose_cam0.camera_translation[0] << ", "
            << fused.pose_cam0.camera_translation[1] << ", "
            << fused.pose_cam0.camera_translation[2] << ")\n";
    }

    return fused;
}

cv::Vec3f StereoHandFuser::TriangulateRoot(
    const cv::Point2f& point0,
    const cv::Point2f& point1) const {
    std::vector<cv::Point2f> points0 = {point0};
    std::vector<cv::Point2f> points1 = {point1};
    std::vector<cv::Point2f> undistorted0;
    std::vector<cv::Point2f> undistorted1;

    cv::undistortPoints(
        points0,
        undistorted0,
        view0_camera_matrix_,
        view0_dist_coeffs_);
    cv::undistortPoints(
        points1,
        undistorted1,
        view1_camera_matrix_,
        view1_dist_coeffs_);

    cv::Matx34d proj0 = cv::Matx34d::eye();
    cv::Matx34d proj1(
        config_.calibration.rotation.at<double>(0, 0), config_.calibration.rotation.at<double>(0, 1), config_.calibration.rotation.at<double>(0, 2), config_.calibration.translation.at<double>(0),
        config_.calibration.rotation.at<double>(1, 0), config_.calibration.rotation.at<double>(1, 1), config_.calibration.rotation.at<double>(1, 2), config_.calibration.translation.at<double>(1),
        config_.calibration.rotation.at<double>(2, 0), config_.calibration.rotation.at<double>(2, 1), config_.calibration.rotation.at<double>(2, 2), config_.calibration.translation.at<double>(2));

    cv::Mat homogeneous;
    cv::triangulatePoints(
        cv::Mat(proj0),
        cv::Mat(proj1),
        undistorted0,
        undistorted1,
        homogeneous);

    cv::Mat homogeneous64;
    homogeneous.convertTo(homogeneous64, CV_64F);
    const double x = homogeneous64.at<double>(0, 0);
    const double y = homogeneous64.at<double>(1, 0);
    const double z = homogeneous64.at<double>(2, 0);
    const double w = homogeneous64.at<double>(3, 0);

    if (config_.verbose_logging) {
        std::cerr
            << "[fusion] triangulate undist0=" << FormatPoint(undistorted0[0])
            << " undist1=" << FormatPoint(undistorted1[0])
            << " homogeneous=(" << x << ", " << y << ", " << z << ", " << w << ")"
            << " type=" << homogeneous.type()
            << "\n";
    }

    if (std::abs(w) < 1e-12) {
        throw std::runtime_error("triangulation produced near-zero homogeneous w");
    }

    return cv::Vec3f(
        static_cast<float>(x / w),
        static_cast<float>(y / w),
        static_cast<float>(z / w));
}

void StereoHandFuser::ProjectToView0(HandPoseResult& pose) const {
    std::vector<cv::Point3f> points3d;
    points3d.reserve(21);
    for (int i = 0; i < 21; ++i) {
        points3d.emplace_back(
            pose.keypoints_3d[i][0] + pose.camera_translation[0],
            pose.keypoints_3d[i][1] + pose.camera_translation[1],
            pose.keypoints_3d[i][2] + pose.camera_translation[2]);
    }

    std::vector<cv::Point2f> projected;
    cv::projectPoints(
        points3d,
        cv::Vec3d(0.0, 0.0, 0.0),
        cv::Vec3d(0.0, 0.0, 0.0),
        view0_camera_matrix_,
        view0_dist_coeffs_,
        projected);

    for (int i = 0; i < 21; ++i) {
        pose.keypoints_2d[i][0] = projected[i].x;
        pose.keypoints_2d[i][1] = projected[i].y;
    }

    const cv::Rect2f bbox = ComputeBBoxFromPoints(pose.keypoints_2d);
    pose.detection.bbox[0] = bbox.x;
    pose.detection.bbox[1] = bbox.y;
    pose.detection.bbox[2] = bbox.x + bbox.width;
    pose.detection.bbox[3] = bbox.y + bbox.height;
}

void StereoHandFuser::UpdateFilterModel(RootFilterState& state, float dt_seconds) const {
    const float dt = ClampDtSeconds(dt_seconds, config_.root_min_dt_seconds, config_.root_max_dt_seconds);
    state.filter.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    const float q = config_.root_process_noise;
    state.filter.processNoiseCov = (cv::Mat_<float>(6, 6) <<
        0.25f * dt4 * q, 0, 0, 0.5f * dt3 * q, 0, 0,
        0, 0.25f * dt4 * q, 0, 0, 0.5f * dt3 * q, 0,
        0, 0, 0.25f * dt4 * q, 0, 0, 0.5f * dt3 * q,
        0.5f * dt3 * q, 0, 0, dt2 * q, 0, 0,
        0, 0.5f * dt3 * q, 0, 0, dt2 * q, 0,
        0, 0, 0.5f * dt3 * q, 0, 0, dt2 * q);
}

cv::Vec3f StereoHandFuser::FilterRoot(
    cv::Vec3f root_joint_cam0,
    bool is_right,
    std::chrono::steady_clock::time_point timestamp) {
    if (!config_.enable_root_kalman) {
        return root_joint_cam0;
    }

    auto& state = root_filters_[is_right ? 1 : 0];
    state.missing_frames = 0;

    if (!state.initialized) {
        UpdateFilterModel(state, config_.root_max_dt_seconds);
        state.filter.statePost.at<float>(0) = root_joint_cam0[0];
        state.filter.statePost.at<float>(1) = root_joint_cam0[1];
        state.filter.statePost.at<float>(2) = root_joint_cam0[2];
        state.filter.statePost.at<float>(3) = 0.0f;
        state.filter.statePost.at<float>(4) = 0.0f;
        state.filter.statePost.at<float>(5) = 0.0f;
        state.filter.statePre = state.filter.statePost.clone();
        state.last_timestamp = timestamp;
        state.initialized = true;
        return root_joint_cam0;
    }

    const float raw_dt_seconds =
        std::chrono::duration<float>(timestamp - state.last_timestamp).count();
    if (raw_dt_seconds > config_.temporal_reset_seconds) {
        state.initialized = false;
        return FilterRoot(root_joint_cam0, is_right, timestamp);
    }

    const float dt_seconds =
        ClampDtSeconds(raw_dt_seconds, config_.root_min_dt_seconds, config_.root_max_dt_seconds);
    UpdateFilterModel(state, dt_seconds);
    state.filter.predict();
    cv::Mat measurement = (cv::Mat_<float>(3, 1) << root_joint_cam0[0], root_joint_cam0[1], root_joint_cam0[2]);
    const cv::Mat corrected = state.filter.correct(measurement);
    state.last_timestamp = timestamp;

    if (config_.verbose_logging) {
        std::cerr
            << "[fusion] handedness=" << (is_right ? "right" : "left")
            << " kalman_dt=" << dt_seconds
            << " raw_dt=" << raw_dt_seconds
            << "\n";
    }

    return cv::Vec3f(
        corrected.at<float>(0),
        corrected.at<float>(1),
        corrected.at<float>(2));
}

void StereoHandFuser::MarkMissing(bool is_right, std::chrono::steady_clock::time_point timestamp) {
    auto& state = root_filters_[is_right ? 1 : 0];
    state.missing_frames += 1;
    const bool exceeded_frame_gap = state.missing_frames > config_.temporal_reset_frames;
    const bool exceeded_time_gap =
        state.initialized
        && std::chrono::duration<float>(timestamp - state.last_timestamp).count() > config_.temporal_reset_seconds;
    if (exceeded_frame_gap || exceeded_time_gap) {
        state.initialized = false;
    }
}
}  // namespace newnewhand
