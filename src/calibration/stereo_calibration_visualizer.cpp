#include "newnewhand/calibration/stereo_calibration_visualizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace newnewhand {

namespace {

cv::Mat EnsureGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

std::vector<cv::Point3f> BuildBoardObjectCorners(const CheckerboardConfig& checkerboard) {
    std::vector<cv::Point3f> corners;
    corners.reserve(static_cast<std::size_t>(checkerboard.inner_corners_cols * checkerboard.inner_corners_rows));
    for (int row = 0; row < checkerboard.inner_corners_rows; ++row) {
        for (int col = 0; col < checkerboard.inner_corners_cols; ++col) {
            corners.emplace_back(
                static_cast<float>(col) * checkerboard.square_size,
                static_cast<float>(row) * checkerboard.square_size,
                0.0f);
        }
    }
    return corners;
}

std::array<cv::Point3f, 4> BuildBoardOutline(const CheckerboardConfig& checkerboard) {
    const float board_width =
        static_cast<float>(checkerboard.inner_corners_cols - 1) * checkerboard.square_size;
    const float board_height =
        static_cast<float>(checkerboard.inner_corners_rows - 1) * checkerboard.square_size;
    return {
        cv::Point3f(0.0f, 0.0f, 0.0f),
        cv::Point3f(board_width, 0.0f, 0.0f),
        cv::Point3f(board_width, board_height, 0.0f),
        cv::Point3f(0.0f, board_height, 0.0f),
    };
}

cv::Vec3f RotatePoint(const cv::Vec3f& point, float yaw_degrees, float pitch_degrees, float roll_degrees) {
    const float yaw = yaw_degrees * static_cast<float>(CV_PI) / 180.0f;
    const float pitch = pitch_degrees * static_cast<float>(CV_PI) / 180.0f;
    const float roll = roll_degrees * static_cast<float>(CV_PI) / 180.0f;

    const float cosy = std::cos(yaw);
    const float siny = std::sin(yaw);
    const float cosp = std::cos(pitch);
    const float sinp = std::sin(pitch);
    const float cosr = std::cos(roll);
    const float sinr = std::sin(roll);

    cv::Vec3f x = point;
    x = cv::Vec3f(cosy * x[0] + siny * x[2], x[1], -siny * x[0] + cosy * x[2]);
    x = cv::Vec3f(x[0], cosp * x[1] - sinp * x[2], sinp * x[1] + cosp * x[2]);
    x = cv::Vec3f(cosr * x[0] - sinr * x[1], sinr * x[0] + cosr * x[1], x[2]);
    return x;
}

bool ProjectPoint(
    const cv::Vec3f& world_point,
    const cv::Vec3f& scene_center,
    const StereoCalibrationViewStyle& style,
    float view_distance,
    float focal,
    cv::Point2f& projected) {
    cv::Vec3f shifted = world_point - scene_center;
    cv::Vec3f camera_point = RotatePoint(shifted, style.yaw_degrees, style.pitch_degrees, style.roll_degrees);
    camera_point[2] += view_distance;
    if (camera_point[2] <= 1e-5f) {
        return false;
    }

    projected = cv::Point2f(
        focal * (camera_point[0] / camera_point[2]) + style.width * 0.5f,
        focal * (camera_point[1] / camera_point[2]) + style.height * 0.5f);
    return true;
}

void DrawSegment(
    cv::Mat& canvas,
    const cv::Vec3f& p0,
    const cv::Vec3f& p1,
    const cv::Vec3f& scene_center,
    const StereoCalibrationViewStyle& style,
    float view_distance,
    float focal,
    const cv::Scalar& color,
    int thickness) {
    cv::Point2f a;
    cv::Point2f b;
    if (!ProjectPoint(p0, scene_center, style, view_distance, focal, a)) {
        return;
    }
    if (!ProjectPoint(p1, scene_center, style, view_distance, focal, b)) {
        return;
    }
    cv::line(canvas, a, b, color, thickness, cv::LINE_AA);
}

void DrawAxes(
    cv::Mat& canvas,
    const cv::Vec3f& origin,
    const cv::Matx33f& rotation,
    float axis_length,
    const cv::Vec3f& scene_center,
    const StereoCalibrationViewStyle& style,
    float view_distance,
    float focal) {
    DrawSegment(canvas, origin, origin + rotation * cv::Vec3f(axis_length, 0.0f, 0.0f), scene_center, style, view_distance, focal, cv::Scalar(30, 30, 230), 2);
    DrawSegment(canvas, origin, origin + rotation * cv::Vec3f(0.0f, axis_length, 0.0f), scene_center, style, view_distance, focal, cv::Scalar(30, 220, 30), 2);
    DrawSegment(canvas, origin, origin + rotation * cv::Vec3f(0.0f, 0.0f, axis_length), scene_center, style, view_distance, focal, cv::Scalar(230, 120, 30), 2);
}

void DrawCameraFrustum(
    cv::Mat& canvas,
    const cv::Vec3f& camera_center,
    const cv::Matx33f& camera_to_world,
    float frustum_scale,
    const cv::Vec3f& scene_center,
    const StereoCalibrationViewStyle& style,
    float view_distance,
    float focal,
    const cv::Scalar& color) {
    const std::array<cv::Vec3f, 4> image_plane = {
        cv::Vec3f(-0.6f, -0.4f, 1.0f),
        cv::Vec3f(0.6f, -0.4f, 1.0f),
        cv::Vec3f(0.6f, 0.4f, 1.0f),
        cv::Vec3f(-0.6f, 0.4f, 1.0f),
    };

    std::array<cv::Vec3f, 4> frustum_corners;
    for (std::size_t i = 0; i < image_plane.size(); ++i) {
        frustum_corners[i] = camera_center + camera_to_world * (image_plane[i] * frustum_scale);
    }

    for (const auto& corner : frustum_corners) {
        DrawSegment(canvas, camera_center, corner, scene_center, style, view_distance, focal, color, 2);
    }
    for (std::size_t i = 0; i < frustum_corners.size(); ++i) {
        const auto& a = frustum_corners[i];
        const auto& b = frustum_corners[(i + 1) % frustum_corners.size()];
        DrawSegment(canvas, a, b, scene_center, style, view_distance, focal, color, 2);
    }

    if (style.draw_axes) {
        DrawAxes(canvas, camera_center, camera_to_world, frustum_scale * 0.5f, scene_center, style, view_distance, focal);
    }
}

}  // namespace

std::vector<StereoCalibrationBoardPose> StereoCalibrationVisualizer::EstimateBoardPoses(
    const StereoCalibrationResult& calibration,
    bool use_find_chessboard_sb,
    std::size_t max_pairs) {
    if (!calibration.success) {
        throw std::invalid_argument("stereo calibration result is not marked as successful");
    }
    if (calibration.observations.empty()) {
        throw std::invalid_argument("stereo calibration result does not contain valid image pairs");
    }

    const cv::Size board_size(
        calibration.checkerboard.inner_corners_cols,
        calibration.checkerboard.inner_corners_rows);
    const std::vector<cv::Point3f> object_corners = BuildBoardObjectCorners(calibration.checkerboard);

    std::vector<StereoCalibrationBoardPose> board_poses;
    const std::size_t limit =
        max_pairs > 0 ? std::min(max_pairs, calibration.observations.size()) : calibration.observations.size();
    board_poses.reserve(limit);

    for (std::size_t pair_index = 0; pair_index < limit; ++pair_index) {
        const auto& observation = calibration.observations[pair_index];
        const cv::Mat left_image = cv::imread(observation.image_pair.left_path.string(), cv::IMREAD_COLOR);
        if (left_image.empty()) {
            continue;
        }

        std::vector<cv::Point2f> left_corners;
        const cv::Mat left_gray = EnsureGrayscale(left_image);
        bool found = false;
        if (use_find_chessboard_sb) {
            #if CV_VERSION_MAJOR >= 4
            const int flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE;
            found = cv::findChessboardCornersSB(left_gray, board_size, left_corners, flags);
            #else
            const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
            found = cv::findChessboardCorners(left_gray, board_size, left_corners, flags);
            if (found) {
                cv::cornerSubPix(
                    left_gray,
                    left_corners,
                    cv::Size(11, 11),
                    cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
            }
            #endif
        } else {
            const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
            found = cv::findChessboardCorners(left_gray, board_size, left_corners, flags);
            if (found) {
                cv::cornerSubPix(
                    left_gray,
                    left_corners,
                    cv::Size(11, 11),
                    cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
            }
        }
        if (!found) {
            continue;
        }

        cv::Mat rvec;
        cv::Mat tvec;
        if (!cv::solvePnP(
                object_corners,
                left_corners,
                calibration.left_camera_matrix,
                calibration.left_dist_coeffs,
                rvec,
                tvec)) {
            continue;
        }

        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);

        StereoCalibrationBoardPose board_pose;
        board_pose.image_pair = observation.image_pair;
        board_pose.rotation_board_to_cam0 = cv::Matx33f(rotation);
        board_pose.translation_board_to_cam0 = cv::Vec3f(tvec);
        board_poses.push_back(board_pose);
    }

    return board_poses;
}

cv::Mat StereoCalibrationVisualizer::RenderScene(
    const StereoCalibrationResult& calibration,
    const std::vector<StereoCalibrationBoardPose>& board_poses,
    const StereoCalibrationViewStyle& style) {
    cv::Mat canvas(style.height, style.width, CV_8UC3, style.background_color);
    if (!calibration.success) {
        return canvas;
    }

    std::vector<cv::Vec3f> scene_points;
    scene_points.emplace_back(0.0f, 0.0f, 0.0f);

    const cv::Matx33f rotation_left_to_right(calibration.rotation);
    const cv::Vec3f translation_left_to_right(calibration.translation);
    const cv::Matx33f cam1_to_world = rotation_left_to_right.t();
    const cv::Vec3f cam1_center = -(cam1_to_world * translation_left_to_right);
    scene_points.push_back(cam1_center);

    const auto board_outline = BuildBoardOutline(calibration.checkerboard);
    for (const auto& board_pose : board_poses) {
        for (const auto& corner : board_outline) {
            scene_points.push_back(board_pose.rotation_board_to_cam0 * cv::Vec3f(corner) + board_pose.translation_board_to_cam0);
        }
    }

    cv::Vec3f scene_center(0.0f, 0.0f, 0.0f);
    for (const auto& point : scene_points) {
        scene_center += point;
    }
    scene_center *= (1.0f / static_cast<float>(scene_points.size()));

    float max_radius = 0.0f;
    for (const auto& point : scene_points) {
        max_radius = std::max(max_radius, static_cast<float>(cv::norm(point - scene_center)));
    }
    if (max_radius < 1e-5f) {
        max_radius = 0.2f;
    }

    const float view_distance = max_radius * (2.0f + style.fit_padding);
    const float focal = static_cast<float>(std::min(style.width, style.height)) * 0.9f;

    DrawCameraFrustum(
        canvas,
        cv::Vec3f(0.0f, 0.0f, 0.0f),
        cv::Matx33f::eye(),
        max_radius * 0.5f,
        scene_center,
        style,
        view_distance,
        focal,
        cv::Scalar(60, 220, 60));
    DrawCameraFrustum(
        canvas,
        cam1_center,
        cam1_to_world,
        max_radius * 0.5f,
        scene_center,
        style,
        view_distance,
        focal,
        cv::Scalar(220, 180, 60));

    for (std::size_t pose_index = 0; pose_index < board_poses.size(); ++pose_index) {
        const auto& board_pose = board_poses[pose_index];
        std::array<cv::Vec3f, 4> board_world;
        for (std::size_t corner_index = 0; corner_index < board_outline.size(); ++corner_index) {
            board_world[corner_index] =
                board_pose.rotation_board_to_cam0 * cv::Vec3f(board_outline[corner_index])
                + board_pose.translation_board_to_cam0;
        }

        const cv::Scalar color = pose_index % 2 == 0 ? cv::Scalar(255, 180, 80) : cv::Scalar(80, 200, 255);
        for (std::size_t corner_index = 0; corner_index < board_world.size(); ++corner_index) {
            DrawSegment(
                canvas,
                board_world[corner_index],
                board_world[(corner_index + 1) % board_world.size()],
                scene_center,
                style,
                view_distance,
                focal,
                color,
                2);
        }
        if (style.draw_axes) {
            DrawAxes(
                canvas,
                board_pose.translation_board_to_cam0,
                board_pose.rotation_board_to_cam0,
                calibration.checkerboard.square_size * 2.0f,
                scene_center,
                style,
                view_distance,
                focal);
        }

        if (style.draw_board_ids) {
            cv::Point2f label_point;
            if (ProjectPoint(
                    board_pose.translation_board_to_cam0,
                    scene_center,
                    style,
                    view_distance,
                    focal,
                    label_point)) {
                cv::putText(
                    canvas,
                    std::to_string(pose_index),
                    label_point,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(235, 235, 235),
                    1);
            }
        }
    }

    cv::putText(
        canvas,
        "cam0 (green), cam1 (yellow), checkerboards",
        cv::Point(18, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(220, 220, 220),
        2);
    return canvas;
}

}  // namespace newnewhand
