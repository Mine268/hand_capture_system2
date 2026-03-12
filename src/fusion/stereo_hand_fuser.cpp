#include "newnewhand/fusion/stereo_hand_fuser.h"

#include <algorithm>
#include <array>
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

}  // namespace

StereoHandFuser::StereoHandFuser(StereoHandFuserConfig config)
    : config_(std::move(config)) {
    if (!config_.calibration.success) {
        throw std::invalid_argument("stereo calibration must be loaded before fusion");
    }
}

StereoFusedHandPoseFrame StereoHandFuser::Fuse(const StereoSingleViewPoseFrame& stereo_frame) const {
    StereoFusedHandPoseFrame fused_frame;
    fused_frame.capture_index = stereo_frame.capture_index;
    fused_frame.trigger_timestamp = stereo_frame.trigger_timestamp;

    for (bool is_right : {false, true}) {
        FusedHandPose fused_hand = FuseHandByHandedness(stereo_frame, is_right);
        if (!fused_hand.has_view0 && !fused_hand.has_view1) {
            continue;
        }
        fused_frame.hands.push_back(std::move(fused_hand));
    }

    std::vector<HandPoseResult> fused_results;
    fused_results.reserve(fused_frame.hands.size());
    for (const auto& hand : fused_frame.hands) {
        fused_results.push_back(hand.pose_cam0);
    }
    fused_frame.fused_third_person_image = RenderThirdPersonHandMeshView(fused_results);
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

    fs << "capture_index" << static_cast<int64_t>(frame.capture_index);
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

FusedHandPose StereoHandFuser::FuseHandByHandedness(const StereoSingleViewPoseFrame& frame, bool is_right) const {
    const HandPoseResult* view0 = FindByHandedness(frame.views[0].hand_poses, is_right);
    const HandPoseResult* view1 = FindByHandedness(frame.views[1].hand_poses, is_right);

    FusedHandPose fused;
    fused.is_right = is_right;
    fused.has_view0 = view0 != nullptr;
    fused.has_view1 = view1 != nullptr;

    if (!view0 && !view1) {
        return fused;
    }

    if (view0 && view1 && config_.prefer_view0_geometry) {
        fused.pose_cam0 = *view0;
        fused.root_joint_cam0 = TriangulateRoot(
            cv::Point2f(view0->keypoints_2d[0][0], view0->keypoints_2d[0][1]),
            cv::Point2f(view1->keypoints_2d[0][0], view1->keypoints_2d[0][1]));
        fused.pose_cam0.camera_translation[0] = fused.root_joint_cam0[0] - fused.pose_cam0.keypoints_3d[0][0];
        fused.pose_cam0.camera_translation[1] = fused.root_joint_cam0[1] - fused.pose_cam0.keypoints_3d[0][1];
        fused.pose_cam0.camera_translation[2] = fused.root_joint_cam0[2] - fused.pose_cam0.keypoints_3d[0][2];
        ProjectToView0(fused.pose_cam0);
        fused.fused_from_stereo = true;
        return fused;
    }

    if (view0 && config_.keep_monocular_fallback) {
        fused.pose_cam0 = *view0;
        fused.root_joint_cam0 = cv::Vec3f(
            fused.pose_cam0.keypoints_3d[0][0] + fused.pose_cam0.camera_translation[0],
            fused.pose_cam0.keypoints_3d[0][1] + fused.pose_cam0.camera_translation[1],
            fused.pose_cam0.keypoints_3d[0][2] + fused.pose_cam0.camera_translation[2]);
        ProjectToView0(fused.pose_cam0);
        return fused;
    }

    if (view1 && config_.keep_monocular_fallback) {
        fused.pose_cam0 = *view1;
        TransformView1PoseToView0(fused.pose_cam0);
        fused.root_joint_cam0 = cv::Vec3f(
            fused.pose_cam0.keypoints_3d[0][0] + fused.pose_cam0.camera_translation[0],
            fused.pose_cam0.keypoints_3d[0][1] + fused.pose_cam0.camera_translation[1],
            fused.pose_cam0.keypoints_3d[0][2] + fused.pose_cam0.camera_translation[2]);
        ProjectToView0(fused.pose_cam0);
        return fused;
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
        config_.calibration.left_camera_matrix,
        config_.calibration.left_dist_coeffs);
    cv::undistortPoints(
        points1,
        undistorted1,
        config_.calibration.right_camera_matrix,
        config_.calibration.right_dist_coeffs);

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

    const double w = homogeneous.at<double>(3, 0);
    return cv::Vec3f(
        static_cast<float>(homogeneous.at<double>(0, 0) / w),
        static_cast<float>(homogeneous.at<double>(1, 0) / w),
        static_cast<float>(homogeneous.at<double>(2, 0) / w));
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
        config_.calibration.left_camera_matrix,
        config_.calibration.left_dist_coeffs,
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

void StereoHandFuser::TransformView1PoseToView0(HandPoseResult& pose) const {
    const cv::Matx33f rotation_01(config_.calibration.rotation);
    const cv::Matx33f rotation_10 = rotation_01.t();
    const cv::Vec3f translation_01(config_.calibration.translation);

    for (int i = 0; i < 21; ++i) {
        cv::Vec3f p(pose.keypoints_3d[i][0], pose.keypoints_3d[i][1], pose.keypoints_3d[i][2]);
        p = rotation_10 * p;
        pose.keypoints_3d[i][0] = p[0];
        pose.keypoints_3d[i][1] = p[1];
        pose.keypoints_3d[i][2] = p[2];
    }
    for (int i = 0; i < 778; ++i) {
        cv::Vec3f p(pose.vertices[i][0], pose.vertices[i][1], pose.vertices[i][2]);
        p = rotation_10 * p;
        pose.vertices[i][0] = p[0];
        pose.vertices[i][1] = p[1];
        pose.vertices[i][2] = p[2];
    }

    cv::Vec3f translation_view1(pose.camera_translation[0], pose.camera_translation[1], pose.camera_translation[2]);
    const cv::Vec3f translation_view0 = rotation_10 * (translation_view1 - translation_01);
    pose.camera_translation[0] = translation_view0[0];
    pose.camera_translation[1] = translation_view0[1];
    pose.camera_translation[2] = translation_view0[2];

    const cv::Matx33f global_rot1 = cv::Matx33f(
        hand_pose_utils::RotationVectorToRotationMatrix(
            cv::Vec3f(pose.global_orient[0], pose.global_orient[1], pose.global_orient[2])));
    const cv::Matx33f global_rot0 = rotation_10 * global_rot1;
    const cv::Vec3f global_rotvec0 = hand_pose_utils::RotationMatrixToRotationVector(global_rot0);
    pose.global_orient[0] = global_rotvec0[0];
    pose.global_orient[1] = global_rotvec0[1];
    pose.global_orient[2] = global_rotvec0[2];
}

}  // namespace newnewhand
