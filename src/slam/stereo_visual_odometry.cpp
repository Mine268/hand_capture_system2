#include "newnewhand/slam/stereo_visual_odometry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace newnewhand {

namespace {

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

cv::Mat ExtractCameraMatrix(const cv::Mat& projection) {
    return projection.colRange(0, 3).clone();
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

cv::Vec3f MatVecMul(const cv::Matx33f& matrix, const cv::Vec3f& v) {
    return cv::Vec3f(
        matrix(0, 0) * v[0] + matrix(0, 1) * v[1] + matrix(0, 2) * v[2],
        matrix(1, 0) * v[0] + matrix(1, 1) * v[1] + matrix(1, 2) * v[2],
        matrix(2, 0) * v[0] + matrix(2, 1) * v[1] + matrix(2, 2) * v[2]);
}

cv::Vec3f ToVec3f(const cv::Mat& vector) {
    cv::Mat vector32f;
    vector.convertTo(vector32f, CV_32F);
    return cv::Vec3f(
        vector32f.at<float>(0, 0),
        vector32f.at<float>(1, 0),
        vector32f.at<float>(2, 0));
}

cv::Point2f ProjectPoint(const cv::Mat& camera_matrix, const cv::Vec3f& point_cam) {
    cv::Mat camera_matrix64f;
    camera_matrix.convertTo(camera_matrix64f, CV_64F);
    const double fx = camera_matrix64f.at<double>(0, 0);
    const double fy = camera_matrix64f.at<double>(1, 1);
    const double cx = camera_matrix64f.at<double>(0, 2);
    const double cy = camera_matrix64f.at<double>(1, 2);
    return cv::Point2f(
        static_cast<float>(fx * point_cam[0] / point_cam[2] + cx),
        static_cast<float>(fy * point_cam[1] / point_cam[2] + cy));
}

float RotationAngleDegrees(const cv::Matx33f& rotation) {
    const float trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    const float cos_angle = std::clamp((trace - 1.0f) * 0.5f, -1.0f, 1.0f);
    return std::acos(cos_angle) * 180.0f / static_cast<float>(CV_PI);
}

struct FrameFeatures {
    std::vector<cv::KeyPoint> left_keypoints;
    cv::Mat left_descriptors;
    std::vector<cv::KeyPoint> stereo_left_keypoints;
    cv::Mat stereo_left_descriptors;
    std::vector<cv::Vec3f> stereo_points_cam0;
    int right_keypoints = 0;
    int stereo_descriptor_pairs = 0;
    int stereo_unique_matches = 0;
    int invalid_triangulation_w = 0;
    int invalid_reprojection = 0;
    int valid_disparity_keypoints = 0;
    int invalid_nonfinite_disparity = 0;
    int invalid_low_disparity = 0;
    int invalid_depth = 0;
    int valid_disparity_pixels = 0;
    float min_valid_disparity = 0.0f;
    float max_valid_disparity = 0.0f;
    float min_depth_cam0 = 0.0f;
    float max_depth_cam0 = 0.0f;
    float mean_depth_cam0 = 0.0f;
    float mean_reprojection_error_px = 0.0f;
};

}  // namespace

struct StereoVisualOdometry::Impl {
    explicit Impl(StereoVisualOdometryConfig config_in)
        : config(std::move(config_in)),
          orb(cv::ORB::create(
              config.max_features,
              config.scale_factor,
              config.num_levels,
              31,
              0,
              2,
              cv::ORB::HARRIS_SCORE,
              31,
              config.fast_threshold)),
          stereo_matcher(cv::StereoSGBM::create(
              0,
              config.stereo_num_disparities,
              config.stereo_block_size)) {
        if (!config.calibration.success) {
            throw std::invalid_argument("stereo calibration must be loaded before visual odometry");
        }

        stereo_matcher->setP1(8 * config.stereo_block_size * config.stereo_block_size);
        stereo_matcher->setP2(32 * config.stereo_block_size * config.stereo_block_size);
        stereo_matcher->setUniquenessRatio(config.stereo_uniqueness_ratio);
        stereo_matcher->setSpeckleWindowSize(config.stereo_speckle_window_size);
        stereo_matcher->setSpeckleRange(config.stereo_speckle_range);
        stereo_matcher->setDisp12MaxDiff(config.stereo_disp12_max_diff);
        stereo_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

        if (config.input_views_are_undistorted) {
            working_left_camera_matrix = BuildUndistortedCameraMatrix(
                config.calibration.left_camera_matrix,
                config.calibration.left_dist_coeffs,
                config.calibration.image_size);
            working_right_camera_matrix = BuildUndistortedCameraMatrix(
                config.calibration.right_camera_matrix,
                config.calibration.right_dist_coeffs,
                config.calibration.image_size);
            working_zero_dist = cv::Mat::zeros(1, 5, CV_64F);
            projection_right_cam0 = cv::Matx34d(
                config.calibration.rotation.at<double>(0, 0),
                config.calibration.rotation.at<double>(0, 1),
                config.calibration.rotation.at<double>(0, 2),
                config.calibration.translation.at<double>(0),
                config.calibration.rotation.at<double>(1, 0),
                config.calibration.rotation.at<double>(1, 1),
                config.calibration.rotation.at<double>(1, 2),
                config.calibration.translation.at<double>(1),
                config.calibration.rotation.at<double>(2, 0),
                config.calibration.rotation.at<double>(2, 1),
                config.calibration.rotation.at<double>(2, 2),
                config.calibration.translation.at<double>(2));
            rotation_right_from_left = ToMatx33f(config.calibration.rotation);
            translation_right_from_left = ToVec3f(config.calibration.translation);
        } else {
            rectification_left_inv = ToMatx33f(config.calibration.rectification_left).t();
            rectified_camera_matrix = ExtractCameraMatrix(config.calibration.projection_left);
            rectified_zero_dist = cv::Mat::zeros(1, 5, CV_64F);

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

        if (config.verbose_logging) {
            std::cerr
                << "[slam] config mode="
                << (config.input_views_are_undistorted ? "undistorted" : "rectified")
                << " max_features=" << config.max_features
                << " fast_threshold=" << config.fast_threshold
                << " levels=" << config.num_levels
                << " ratio=" << config.temporal_ratio_test
                << " min_stereo=" << config.min_stereo_points
                << " min_track=" << config.min_tracking_points
                << " sgbm_disp=" << config.stereo_num_disparities
                << " sgbm_block=" << config.stereo_block_size
                << " pnp_iter=" << config.pnp_iterations
                << " pnp_reproj=" << config.pnp_reprojection_error_px
                << "\n";
        }
    }

    void Reset() {
        initialized = false;
        previous_descriptors.release();
        previous_world_points.clear();
        trajectory_world.clear();
        current_camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
        current_rotation_world_from_cam0 = cv::Matx33f::eye();
    }

    FrameFeatures ExtractFeaturesRectified(const cv::Mat& left_rectified, const cv::Mat& right_rectified) {
        FrameFeatures features;
        cv::Mat left_gray;
        cv::Mat right_gray;
        if (left_rectified.channels() == 1) {
            left_gray = left_rectified;
        } else {
            cv::cvtColor(left_rectified, left_gray, cv::COLOR_BGR2GRAY);
        }
        if (right_rectified.channels() == 1) {
            right_gray = right_rectified;
        } else {
            cv::cvtColor(right_rectified, right_gray, cv::COLOR_BGR2GRAY);
        }
        orb->detectAndCompute(
            left_gray,
            cv::noArray(),
            features.left_keypoints,
            features.left_descriptors);
        if (features.left_descriptors.empty()) {
            return features;
        }

        cv::Mat disparity_16s;
        stereo_matcher->compute(left_gray, right_gray, disparity_16s);

        cv::Mat disparity_32f;
        disparity_16s.convertTo(disparity_32f, CV_32F, 1.0 / 16.0);

        bool has_valid_disparity = false;
        for (int y = 0; y < disparity_32f.rows; ++y) {
            for (int x = 0; x < disparity_32f.cols; ++x) {
                const float disparity = disparity_32f.at<float>(y, x);
                if (!std::isfinite(disparity) || disparity < config.min_disparity_px) {
                    continue;
                }
                features.valid_disparity_pixels += 1;
                if (!has_valid_disparity) {
                    features.min_valid_disparity = disparity;
                    features.max_valid_disparity = disparity;
                    has_valid_disparity = true;
                } else {
                    features.min_valid_disparity = std::min(features.min_valid_disparity, disparity);
                    features.max_valid_disparity = std::max(features.max_valid_disparity, disparity);
                }
            }
        }

        cv::Mat points_3d_rectified;
        cv::reprojectImageTo3D(
            disparity_32f,
            points_3d_rectified,
            config.calibration.disparity_to_depth,
            false,
            CV_32F);

        for (int keypoint_index = 0; keypoint_index < static_cast<int>(features.left_keypoints.size()); ++keypoint_index) {
            const cv::Point2f& point = features.left_keypoints[keypoint_index].pt;
            const int x = static_cast<int>(std::lround(point.x));
            const int y = static_cast<int>(std::lround(point.y));
            if (x < 0 || y < 0 || x >= disparity_32f.cols || y >= disparity_32f.rows) {
                continue;
            }

            const float disparity = disparity_32f.at<float>(y, x);
            if (!std::isfinite(disparity)) {
                features.invalid_nonfinite_disparity += 1;
                continue;
            }
            if (disparity < config.min_disparity_px) {
                features.invalid_low_disparity += 1;
                continue;
            }
            features.valid_disparity_keypoints += 1;

            const cv::Vec3f point_rectified = points_3d_rectified.at<cv::Vec3f>(y, x);
            if (!std::isfinite(point_rectified[0])
                || !std::isfinite(point_rectified[1])
                || !std::isfinite(point_rectified[2])
                || point_rectified[2] <= 0.0f
                || point_rectified[2] > 10.0f) {
                features.invalid_depth += 1;
                continue;
            }

            features.stereo_left_keypoints.push_back(features.left_keypoints[keypoint_index]);
            features.stereo_left_descriptors.push_back(features.left_descriptors.row(keypoint_index));
            features.stereo_points_cam0.push_back(MatVecMul(rectification_left_inv, point_rectified));
        }

        features.stereo_left_descriptors = features.stereo_left_descriptors.clone();
        return features;
    }

    FrameFeatures ExtractFeaturesUndistorted(const cv::Mat& left_undistorted, const cv::Mat& right_undistorted) {
        FrameFeatures features;
        cv::Mat left_gray;
        cv::Mat right_gray;
        if (left_undistorted.channels() == 1) {
            left_gray = left_undistorted;
        } else {
            cv::cvtColor(left_undistorted, left_gray, cv::COLOR_BGR2GRAY);
        }
        if (right_undistorted.channels() == 1) {
            right_gray = right_undistorted;
        } else {
            cv::cvtColor(right_undistorted, right_gray, cv::COLOR_BGR2GRAY);
        }

        std::vector<cv::KeyPoint> right_keypoints;
        cv::Mat right_descriptors;
        orb->detectAndCompute(
            left_gray,
            cv::noArray(),
            features.left_keypoints,
            features.left_descriptors);
        orb->detectAndCompute(
            right_gray,
            cv::noArray(),
            right_keypoints,
            right_descriptors);
        features.right_keypoints = static_cast<int>(right_keypoints.size());
        if (features.left_descriptors.empty() || right_descriptors.empty()) {
            return features;
        }

        cv::BFMatcher stereo_descriptor_matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> stereo_knn_matches;
        stereo_descriptor_matcher.knnMatch(features.left_descriptors, right_descriptors, stereo_knn_matches, 2);
        features.stereo_descriptor_pairs = static_cast<int>(stereo_knn_matches.size());

        std::unordered_map<int, cv::DMatch> unique_matches;
        unique_matches.reserve(stereo_knn_matches.size());
        for (const auto& pair : stereo_knn_matches) {
            if (pair.size() < 2) {
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            if (best.distance >= config.temporal_ratio_test * second.distance) {
                continue;
            }
            auto it = unique_matches.find(best.trainIdx);
            if (it == unique_matches.end() || best.distance < it->second.distance) {
                unique_matches[best.trainIdx] = best;
            }
        }
        features.stereo_unique_matches = static_cast<int>(unique_matches.size());

        const cv::Matx34d projection_left_cam0 = cv::Matx34d::eye();
        bool has_valid_depth = false;
        float depth_sum = 0.0f;
        float reprojection_sum = 0.0f;
        for (const auto& [_, match] : unique_matches) {
            if (match.queryIdx < 0
                || match.queryIdx >= static_cast<int>(features.left_keypoints.size())
                || match.trainIdx < 0
                || match.trainIdx >= static_cast<int>(right_keypoints.size())) {
                continue;
            }

            const cv::Point2f left_point = features.left_keypoints[match.queryIdx].pt;
            const cv::Point2f right_point = right_keypoints[match.trainIdx].pt;
            std::vector<cv::Point2f> left_points = {left_point};
            std::vector<cv::Point2f> right_points = {right_point};
            std::vector<cv::Point2f> normalized_left;
            std::vector<cv::Point2f> normalized_right;
            cv::undistortPoints(
                left_points,
                normalized_left,
                working_left_camera_matrix,
                working_zero_dist);
            cv::undistortPoints(
                right_points,
                normalized_right,
                working_right_camera_matrix,
                working_zero_dist);

            cv::Mat homogeneous;
            cv::triangulatePoints(
                cv::Mat(projection_left_cam0),
                cv::Mat(projection_right_cam0),
                normalized_left,
                normalized_right,
                homogeneous);

            cv::Mat homogeneous64f;
            homogeneous.convertTo(homogeneous64f, CV_64F);
            const double w = homogeneous64f.at<double>(3, 0);
            if (!std::isfinite(w) || std::abs(w) < 1e-12) {
                features.invalid_triangulation_w += 1;
                continue;
            }

            const cv::Vec3f point_cam0(
                static_cast<float>(homogeneous64f.at<double>(0, 0) / w),
                static_cast<float>(homogeneous64f.at<double>(1, 0) / w),
                static_cast<float>(homogeneous64f.at<double>(2, 0) / w));
            if (!std::isfinite(point_cam0[0])
                || !std::isfinite(point_cam0[1])
                || !std::isfinite(point_cam0[2])
                || point_cam0[2] <= 0.0f
                || point_cam0[2] > 10.0f) {
                features.invalid_depth += 1;
                continue;
            }

            const cv::Vec3f point_cam1 = MatVecMul(rotation_right_from_left, point_cam0) + translation_right_from_left;
            if (point_cam1[2] <= 0.0f || !std::isfinite(point_cam1[2])) {
                features.invalid_depth += 1;
                continue;
            }

            const cv::Point2f reprojected_left = ProjectPoint(working_left_camera_matrix, point_cam0);
            const cv::Point2f reprojected_right = ProjectPoint(working_right_camera_matrix, point_cam1);
            const float reprojection_error =
                cv::norm(reprojected_left - left_point) + cv::norm(reprojected_right - right_point);
            if (!std::isfinite(reprojection_error)
                || reprojection_error > 2.0f * config.pnp_reprojection_error_px) {
                features.invalid_reprojection += 1;
                continue;
            }

            features.valid_disparity_keypoints += 1;
            features.stereo_left_keypoints.push_back(features.left_keypoints[match.queryIdx]);
            features.stereo_left_descriptors.push_back(features.left_descriptors.row(match.queryIdx));
            features.stereo_points_cam0.push_back(point_cam0);
            if (!has_valid_depth) {
                features.min_depth_cam0 = point_cam0[2];
                features.max_depth_cam0 = point_cam0[2];
                has_valid_depth = true;
            } else {
                features.min_depth_cam0 = std::min(features.min_depth_cam0, point_cam0[2]);
                features.max_depth_cam0 = std::max(features.max_depth_cam0, point_cam0[2]);
            }
            depth_sum += point_cam0[2];
            reprojection_sum += reprojection_error;
        }

        if (!features.stereo_points_cam0.empty()) {
            features.mean_depth_cam0 = depth_sum / static_cast<float>(features.stereo_points_cam0.size());
            features.mean_reprojection_error_px =
                reprojection_sum / static_cast<float>(features.stereo_points_cam0.size());
        }
        features.stereo_left_descriptors = features.stereo_left_descriptors.clone();
        return features;
    }

    FrameFeatures ExtractFeatures(const cv::Mat& left_image, const cv::Mat& right_image) {
        return config.input_views_are_undistorted
            ? ExtractFeaturesUndistorted(left_image, right_image)
            : ExtractFeaturesRectified(left_image, right_image);
    }

    StereoCameraTrackingResult InitializeFromFeatures(
        const StereoSingleViewPoseFrame& frame,
        const FrameFeatures& features,
        bool reinitialized) {
        StereoCameraTrackingResult result;
        result.capture_index = frame.capture_index;
        result.trigger_timestamp = frame.trigger_timestamp;
        result.initialized = false;
        result.tracking_ok = false;
        result.reinitialized = reinitialized;
        result.left_keypoints = static_cast<int>(features.left_keypoints.size());
        result.stereo_points = static_cast<int>(features.stereo_points_cam0.size());
        result.valid_disparity_keypoints = features.valid_disparity_keypoints;
        result.invalid_nonfinite_disparity = features.invalid_nonfinite_disparity;
        result.invalid_low_disparity = features.invalid_low_disparity;
        result.invalid_depth = features.invalid_depth;
        result.valid_disparity_pixels = features.valid_disparity_pixels;
        result.min_valid_disparity = features.min_valid_disparity;
        result.max_valid_disparity = features.max_valid_disparity;

        if (features.stereo_points_cam0.size() < static_cast<std::size_t>(config.min_stereo_points)) {
            result.status_message = "not enough stereo correspondences";
            if (config.verbose_logging) {
                std::cerr
                    << "[slam] init pending capture=" << frame.capture_index
                    << " mode=" << (config.input_views_are_undistorted ? "undistorted" : "rectified")
                    << " left_kps=" << result.left_keypoints
                    << " right_kps=" << features.right_keypoints
                    << " stereo_pairs=" << features.stereo_descriptor_pairs
                    << " stereo_unique=" << features.stereo_unique_matches
                    << " stereo_points=" << result.stereo_points
                    << " valid_disp_kp=" << result.valid_disparity_keypoints
                    << " invalid_nan=" << result.invalid_nonfinite_disparity
                    << " invalid_low=" << result.invalid_low_disparity
                    << " invalid_depth=" << result.invalid_depth
                    << " invalid_w=" << features.invalid_triangulation_w
                    << " invalid_reproj=" << features.invalid_reprojection
                    << " valid_disp_px=" << result.valid_disparity_pixels
                    << " disp_range=[" << result.min_valid_disparity
                    << ", " << result.max_valid_disparity << "]"
                    << " depth_range=[" << features.min_depth_cam0
                    << ", " << features.max_depth_cam0 << "]"
                    << " depth_mean=" << features.mean_depth_cam0
                    << " reproj_mean=" << features.mean_reprojection_error_px
                    << " required=" << config.min_stereo_points
                    << "\n";
            }
            return result;
        }

        previous_descriptors = features.stereo_left_descriptors.clone();
        previous_world_points.clear();
        previous_world_points.reserve(features.stereo_points_cam0.size());
        for (const auto& point_cam0 : features.stereo_points_cam0) {
            previous_world_points.emplace_back(point_cam0[0], point_cam0[1], point_cam0[2]);
        }

        initialized = true;
        trajectory_world.clear();
        trajectory_world.push_back(cv::Vec3f(0.0f, 0.0f, 0.0f));
        current_camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
        current_rotation_world_from_cam0 = cv::Matx33f::eye();

        result.initialized = true;
        result.tracking_ok = true;
        result.rotation_world_from_cam0 = current_rotation_world_from_cam0;
        result.camera_center_world = current_camera_center_world;
        result.trajectory_world = trajectory_world;
        result.status_message = reinitialized ? "tracking reinitialized" : "tracking initialized";
        if (config.verbose_logging) {
            std::cerr
                << "[slam] "
                << result.status_message
                << " capture=" << frame.capture_index
                << " mode=" << (config.input_views_are_undistorted ? "undistorted" : "rectified")
                << " left_kps=" << result.left_keypoints
                << " right_kps=" << features.right_keypoints
                << " stereo_pairs=" << features.stereo_descriptor_pairs
                << " stereo_unique=" << features.stereo_unique_matches
                << " stereo_points=" << result.stereo_points
                << " valid_disp_kp=" << result.valid_disparity_keypoints
                << " invalid_depth=" << result.invalid_depth
                << " invalid_w=" << features.invalid_triangulation_w
                << " invalid_reproj=" << features.invalid_reprojection
                << " depth_range=[" << features.min_depth_cam0
                << ", " << features.max_depth_cam0 << "]"
                << " depth_mean=" << features.mean_depth_cam0
                << " reproj_mean=" << features.mean_reprojection_error_px
                << " valid_disp_px=" << result.valid_disparity_pixels
                << "\n";
        }
        return result;
    }

    StereoCameraTrackingResult Track(const StereoSingleViewPoseFrame& frame) {
        StereoCameraTrackingResult result;
        result.capture_index = frame.capture_index;
        result.trigger_timestamp = frame.trigger_timestamp;
        result.rotation_world_from_cam0 = current_rotation_world_from_cam0;
        result.camera_center_world = current_camera_center_world;
        result.trajectory_world = trajectory_world;

        if (!frame.is_complete()) {
            result.initialized = initialized;
            result.status_message = "stereo frame incomplete";
            return result;
        }

        if (frame.views[0].camera_frame.bgr_image.empty() || frame.views[1].camera_frame.bgr_image.empty()) {
            result.initialized = initialized;
            result.status_message = "stereo images missing";
            return result;
        }

        cv::Mat left_working;
        cv::Mat right_working;
        if (config.input_views_are_undistorted) {
            left_working = frame.views[0].camera_frame.bgr_image;
            right_working = frame.views[1].camera_frame.bgr_image;
        } else {
            cv::remap(
                frame.views[0].camera_frame.bgr_image,
                left_working,
                left_map_x,
                left_map_y,
                cv::INTER_LINEAR);
            cv::remap(
                frame.views[1].camera_frame.bgr_image,
                right_working,
                right_map_x,
                right_map_y,
                cv::INTER_LINEAR);
        }

        FrameFeatures features = ExtractFeatures(left_working, right_working);
        result.left_keypoints = static_cast<int>(features.left_keypoints.size());
        result.stereo_points = static_cast<int>(features.stereo_points_cam0.size());
        result.valid_disparity_keypoints = features.valid_disparity_keypoints;
        result.invalid_nonfinite_disparity = features.invalid_nonfinite_disparity;
        result.invalid_low_disparity = features.invalid_low_disparity;
        result.invalid_depth = features.invalid_depth;
        result.valid_disparity_pixels = features.valid_disparity_pixels;
        result.min_valid_disparity = features.min_valid_disparity;
        result.max_valid_disparity = features.max_valid_disparity;

        if (!initialized) {
            return InitializeFromFeatures(frame, features, false);
        }

        if (previous_descriptors.empty() || previous_world_points.empty()) {
            Reset();
            return InitializeFromFeatures(frame, features, true);
        }

        std::vector<std::vector<cv::DMatch>> temporal_knn_matches;
        if (!features.left_descriptors.empty()) {
            matcher.knnMatch(previous_descriptors, features.left_descriptors, temporal_knn_matches, 2);
        }

        std::unordered_map<int, cv::DMatch> unique_matches;
        unique_matches.reserve(temporal_knn_matches.size());
        for (const auto& pair : temporal_knn_matches) {
            if (pair.size() < 2) {
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            if (best.distance >= config.temporal_ratio_test * second.distance) {
                continue;
            }
            auto it = unique_matches.find(best.trainIdx);
            if (it == unique_matches.end() || best.distance < it->second.distance) {
                unique_matches[best.trainIdx] = best;
            }
        }

        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        object_points.reserve(unique_matches.size());
        image_points.reserve(unique_matches.size());
        for (const auto& [_, match] : unique_matches) {
            if (match.queryIdx < 0
                || match.queryIdx >= static_cast<int>(previous_world_points.size())
                || match.trainIdx < 0
                || match.trainIdx >= static_cast<int>(features.left_keypoints.size())) {
                continue;
            }
            object_points.push_back(previous_world_points[match.queryIdx]);
            image_points.push_back(features.left_keypoints[match.trainIdx].pt);
        }

        result.matched_points = static_cast<int>(object_points.size());
        if (object_points.size() < static_cast<std::size_t>(config.min_tracking_points)) {
            Reset();
            return InitializeFromFeatures(frame, features, true);
        }

        const cv::Mat& pnp_camera_matrix =
            config.input_views_are_undistorted ? working_left_camera_matrix : rectified_camera_matrix;
        const cv::Mat& pnp_zero_dist =
            config.input_views_are_undistorted ? working_zero_dist : rectified_zero_dist;

        cv::Mat rvec;
        cv::Mat tvec;
        cv::Mat inliers;
        const bool pnp_ok = cv::solvePnPRansac(
            object_points,
            image_points,
            pnp_camera_matrix,
            pnp_zero_dist,
            rvec,
            tvec,
            false,
            config.pnp_iterations,
            config.pnp_reprojection_error_px,
            config.pnp_confidence,
            inliers,
            cv::SOLVEPNP_EPNP);

        if (!pnp_ok || inliers.rows < config.min_tracking_points) {
            if (config.verbose_logging) {
                std::cerr
                    << "[slam] tracking lost capture=" << frame.capture_index
                    << " mode=" << (config.input_views_are_undistorted ? "undistorted" : "rectified")
                    << " left_kps=" << result.left_keypoints
                    << " right_kps=" << features.right_keypoints
                    << " stereo_pairs=" << features.stereo_descriptor_pairs
                    << " stereo_unique=" << features.stereo_unique_matches
                    << " stereo_points=" << result.stereo_points
                    << " matched=" << object_points.size()
                    << " inliers=" << inliers.rows
                    << " valid_disp_kp=" << result.valid_disparity_keypoints
                    << " invalid_nan=" << result.invalid_nonfinite_disparity
                    << " invalid_low=" << result.invalid_low_disparity
                    << " invalid_depth=" << result.invalid_depth
                    << " invalid_w=" << features.invalid_triangulation_w
                    << " invalid_reproj=" << features.invalid_reprojection
                    << " depth_range=[" << features.min_depth_cam0
                    << ", " << features.max_depth_cam0 << "]"
                    << " depth_mean=" << features.mean_depth_cam0
                    << " reproj_mean=" << features.mean_reprojection_error_px
                    << " pnp_model=" << (config.input_views_are_undistorted ? "undistorted_cam0" : "rectified")
                    << "\n";
            }
            Reset();
            return InitializeFromFeatures(frame, features, true);
        }

        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        const cv::Matx33f rotation_current_cam0_from_world = ToMatx33f(rotation_matrix);
        const cv::Matx33f rotation_world_from_cam0 = rotation_current_cam0_from_world.t();
        const cv::Vec3f translation_current_cam0_from_world = ToVec3f(tvec);
        const cv::Vec3f camera_center_world =
            -MatVecMul(rotation_world_from_cam0, translation_current_cam0_from_world);

        std::vector<cv::Point3f> current_world_points;
        current_world_points.reserve(features.stereo_points_cam0.size());
        for (const auto& point_cam0 : features.stereo_points_cam0) {
            const cv::Vec3f point_world =
                MatVecMul(rotation_world_from_cam0, point_cam0) + camera_center_world;
            current_world_points.emplace_back(point_world[0], point_world[1], point_world[2]);
        }

        previous_descriptors = features.stereo_left_descriptors.clone();
        previous_world_points = std::move(current_world_points);
        current_camera_center_world = camera_center_world;
        current_rotation_world_from_cam0 = rotation_world_from_cam0;
        trajectory_world.push_back(camera_center_world);

        result.initialized = true;
        result.tracking_ok = true;
        result.tracking_inliers = inliers.rows;
        result.rotation_world_from_cam0 = current_rotation_world_from_cam0;
        result.camera_center_world = current_camera_center_world;
        result.trajectory_world = trajectory_world;
        result.status_message = "tracking ok";

        if (config.verbose_logging) {
            std::cerr
                << "[slam] capture=" << frame.capture_index
                << " mode=" << (config.input_views_are_undistorted ? "undistorted" : "rectified")
                << " left_kps=" << result.left_keypoints
                << " right_kps=" << features.right_keypoints
                << " stereo_pairs=" << features.stereo_descriptor_pairs
                << " stereo_unique=" << features.stereo_unique_matches
                << " stereo_points=" << result.stereo_points
                << " matched=" << result.matched_points
                << " inliers=" << result.tracking_inliers
                << " valid_disp_kp=" << result.valid_disparity_keypoints
                << " invalid_depth=" << result.invalid_depth
                << " invalid_w=" << features.invalid_triangulation_w
                << " invalid_reproj=" << features.invalid_reprojection
                << " valid_disp_px=" << result.valid_disparity_pixels
                << " disp_range=[" << result.min_valid_disparity
                << ", " << result.max_valid_disparity << "]"
                << " depth_range=[" << features.min_depth_cam0
                << ", " << features.max_depth_cam0 << "]"
                << " depth_mean=" << features.mean_depth_cam0
                << " reproj_mean=" << features.mean_reprojection_error_px
                << " pnp_model=" << (config.input_views_are_undistorted ? "undistorted_cam0" : "rectified")
                << " center_world=("
                << result.camera_center_world[0] << ", "
                << result.camera_center_world[1] << ", "
                << result.camera_center_world[2] << ")\n";
        }

        return result;
    }

    StereoVisualOdometryConfig config;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::StereoSGBM> stereo_matcher;
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, false);
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;
    cv::Mat rectified_camera_matrix;
    cv::Mat rectified_zero_dist;
    cv::Mat working_left_camera_matrix;
    cv::Mat working_right_camera_matrix;
    cv::Mat working_zero_dist;
    cv::Mat previous_descriptors;
    cv::Matx34d projection_right_cam0 = cv::Matx34d::eye();
    cv::Matx33f rectification_left_inv = cv::Matx33f::eye();
    cv::Matx33f rotation_right_from_left = cv::Matx33f::eye();
    cv::Matx33f current_rotation_world_from_cam0 = cv::Matx33f::eye();
    cv::Vec3f current_camera_center_world = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec3f translation_right_from_left = cv::Vec3f(0.0f, 0.0f, 0.0f);
    std::vector<cv::Point3f> previous_world_points;
    std::vector<cv::Vec3f> trajectory_world;
    bool initialized = false;
};

StereoVisualOdometry::StereoVisualOdometry(StereoVisualOdometryConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

StereoVisualOdometry::~StereoVisualOdometry() = default;

StereoVisualOdometry::StereoVisualOdometry(StereoVisualOdometry&&) noexcept = default;

StereoVisualOdometry& StereoVisualOdometry::operator=(StereoVisualOdometry&&) noexcept = default;

StereoCameraTrackingResult StereoVisualOdometry::Track(const StereoSingleViewPoseFrame& frame) {
    return impl_->Track(frame);
}

void StereoVisualOdometry::Reset() {
    impl_->Reset();
}

}  // namespace newnewhand
