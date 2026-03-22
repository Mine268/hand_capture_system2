#include "newnewhand/slam/offline_camera_trajectory_optimizer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace newnewhand {

namespace {

using InformationMatrix = Eigen::Matrix<double, 6, 6>;

g2o::Isometry3 ToIsometry3(const cv::Matx33f& rotation_world_from_cam0, const cv::Vec3f& camera_center_world) {
    g2o::Isometry3 pose = g2o::Isometry3::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            pose.linear()(r, c) = rotation_world_from_cam0(r, c);
        }
        pose.translation()(r) = camera_center_world[r];
    }
    return pose;
}

cv::Matx33f ToMatx33f(const g2o::Isometry3& pose) {
    cv::Matx33f rotation = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotation(r, c) = static_cast<float>(pose.linear()(r, c));
        }
    }
    return rotation;
}

cv::Vec3f ToVec3f(const g2o::Isometry3& pose) {
    return cv::Vec3f(
        static_cast<float>(pose.translation()(0)),
        static_cast<float>(pose.translation()(1)),
        static_cast<float>(pose.translation()(2)));
}

InformationMatrix BuildInformationMatrix(double translation_sigma_m, double rotation_sigma_deg) {
    const double translation_weight = 1.0 / std::max(1e-12, translation_sigma_m * translation_sigma_m);
    const double rotation_sigma_rad = rotation_sigma_deg * M_PI / 180.0;
    const double rotation_weight = 1.0 / std::max(1e-12, rotation_sigma_rad * rotation_sigma_rad);

    InformationMatrix information = InformationMatrix::Zero();
    information.block<3, 3>(0, 0).diagonal().array() = translation_weight;
    information.block<3, 3>(3, 3).diagonal().array() = rotation_weight;
    return information;
}

InformationMatrix BuildCharucoInformationMatrix(
    const OfflineCameraTrajectoryOptimizerConfig& config,
    int num_corners,
    float reprojection_error_px) {
    InformationMatrix information =
        BuildInformationMatrix(config.charuco_translation_sigma_m, config.charuco_rotation_sigma_deg);
    const double corners_scale = std::clamp(static_cast<double>(num_corners) / 12.0, 0.5, 2.0);
    const double reproj_scale = 1.0 / std::clamp(static_cast<double>(reprojection_error_px), 0.5, 5.0);
    information *= corners_scale * reproj_scale;
    return information;
}

bool IsUsableCharucoObservation(
    const OfflineCameraTrajectoryOptimizerConfig& config,
    const OfflineCameraTrajectorySample& sample) {
    return sample.has_charuco_pose
        && sample.charuco_num_corners >= config.min_charuco_corners
        && sample.charuco_reprojection_error_px > 0.0f
        && sample.charuco_reprojection_error_px <= config.max_charuco_reprojection_error_px;
}

bool IsUsableSlamPose(const OfflineCameraTrajectorySample& sample) {
    return sample.slam_tracking_result.initialized && sample.slam_tracking_result.tracking_ok;
}

}  // namespace

OfflineCameraTrajectoryOptimizer::OfflineCameraTrajectoryOptimizer(OfflineCameraTrajectoryOptimizerConfig config)
    : config_(std::move(config)) {}

OfflineCameraTrajectoryOptimizationResult OfflineCameraTrajectoryOptimizer::Optimize(
    const std::vector<OfflineCameraTrajectorySample>& samples) const {
    OfflineCameraTrajectoryOptimizationResult result;
    result.optimized_tracking_results.reserve(samples.size());
    if (samples.empty()) {
        return result;
    }

    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    auto linear_solver = std::make_unique<g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>>();
    auto block_solver = std::make_unique<BlockSolver>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(config_.verbose_logging);

    std::vector<g2o::VertexSE3*> vertices(samples.size(), nullptr);
    std::vector<bool> vertex_has_factor(samples.size(), false);

    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const auto& sample = samples[sample_index];
        if (!sample.initial_tracking_result.initialized && !IsUsableCharucoObservation(config_, sample)) {
            continue;
        }

        auto* vertex = new g2o::VertexSE3();
        vertex->setId(static_cast<int>(sample_index));
        if (sample.initial_tracking_result.initialized) {
            vertex->setEstimate(ToIsometry3(
                sample.initial_tracking_result.rotation_world_from_cam0,
                sample.initial_tracking_result.camera_center_world));
        } else {
            vertex->setEstimate(ToIsometry3(
                sample.charuco_rotation_world_from_cam0,
                sample.charuco_camera_center_world));
        }
        optimizer.addVertex(vertex);
        vertices[sample_index] = vertex;
        result.num_vertices += 1;
    }

    const InformationMatrix slam_information =
        BuildInformationMatrix(config_.slam_translation_sigma_m, config_.slam_rotation_sigma_deg);

    std::size_t first_connected_vertex = samples.size();
    bool has_absolute_constraint = false;
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        if (!vertices[sample_index]) {
            continue;
        }
        if (first_connected_vertex == samples.size()) {
            first_connected_vertex = sample_index;
        }

        const auto& sample = samples[sample_index];
        if (IsUsableCharucoObservation(config_, sample)) {
            auto* world_anchor = optimizer.vertex(-1)
                ? static_cast<g2o::VertexSE3*>(optimizer.vertex(-1))
                : nullptr;
            if (!world_anchor) {
                world_anchor = new g2o::VertexSE3();
                world_anchor->setId(-1);
                world_anchor->setEstimate(g2o::Isometry3::Identity());
                world_anchor->setFixed(true);
                optimizer.addVertex(world_anchor);
            }

            auto* edge = new g2o::EdgeSE3();
            edge->setVertex(0, world_anchor);
            edge->setVertex(1, vertices[sample_index]);
            edge->setMeasurement(ToIsometry3(
                sample.charuco_rotation_world_from_cam0,
                sample.charuco_camera_center_world));
            edge->setInformation(BuildCharucoInformationMatrix(
                config_,
                sample.charuco_num_corners,
                sample.charuco_reprojection_error_px));
            optimizer.addEdge(edge);
            vertex_has_factor[sample_index] = true;
            has_absolute_constraint = true;
            result.num_charuco_priors += 1;
        }
    }

    for (std::size_t sample_index = 1; sample_index < samples.size(); ++sample_index) {
        const auto& prev_sample = samples[sample_index - 1];
        const auto& curr_sample = samples[sample_index];
        if (!vertices[sample_index - 1] || !vertices[sample_index]) {
            continue;
        }
        if (!IsUsableSlamPose(prev_sample) || !IsUsableSlamPose(curr_sample) || curr_sample.slam_tracking_result.reinitialized) {
            continue;
        }

        const g2o::Isometry3 prev_pose = ToIsometry3(
            prev_sample.slam_tracking_result.rotation_world_from_cam0,
            prev_sample.slam_tracking_result.camera_center_world);
        const g2o::Isometry3 curr_pose = ToIsometry3(
            curr_sample.slam_tracking_result.rotation_world_from_cam0,
            curr_sample.slam_tracking_result.camera_center_world);

        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, vertices[sample_index - 1]);
        edge->setVertex(1, vertices[sample_index]);
        edge->setMeasurement(prev_pose.inverse() * curr_pose);
        edge->setInformation(slam_information);
        optimizer.addEdge(edge);
        vertex_has_factor[sample_index - 1] = true;
        vertex_has_factor[sample_index] = true;
        result.num_slam_edges += 1;
    }

    if (!has_absolute_constraint && config_.anchor_first_pose_if_unconstrained && first_connected_vertex < samples.size()) {
        vertices[first_connected_vertex]->setFixed(true);
        result.used_fallback_anchor = true;
    }

    if (result.num_vertices == 0 || (result.num_slam_edges == 0 && result.num_charuco_priors == 0)) {
        result.optimized_tracking_results.reserve(samples.size());
        std::vector<cv::Vec3f> trajectory_world;
        for (const auto& sample : samples) {
            auto tracking_result = sample.initial_tracking_result;
            if (tracking_result.initialized) {
                trajectory_world.push_back(tracking_result.camera_center_world);
                tracking_result.trajectory_world = trajectory_world;
            }
            result.optimized_tracking_results.push_back(std::move(tracking_result));
        }
        return result;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(50);

    std::vector<cv::Vec3f> optimized_trajectory;
    optimized_trajectory.reserve(samples.size());
    result.optimized_tracking_results.reserve(samples.size());
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        auto tracking_result = samples[sample_index].initial_tracking_result;
        if (vertices[sample_index] && vertex_has_factor[sample_index]) {
            const auto optimized_pose = vertices[sample_index]->estimate();
            tracking_result.initialized = true;
            tracking_result.tracking_ok = true;
            tracking_result.rotation_world_from_cam0 = ToMatx33f(optimized_pose);
            tracking_result.camera_center_world = ToVec3f(optimized_pose);
            if (IsUsableCharucoObservation(config_, samples[sample_index]) && IsUsableSlamPose(samples[sample_index])) {
                tracking_result.status_message = "optimized pose (charuco + slam)";
            } else if (IsUsableCharucoObservation(config_, samples[sample_index])) {
                tracking_result.status_message = "optimized pose (charuco)";
            } else if (IsUsableSlamPose(samples[sample_index])) {
                tracking_result.status_message = "optimized pose (slam)";
            } else {
                tracking_result.status_message = "optimized pose";
            }
            optimized_trajectory.push_back(tracking_result.camera_center_world);
            tracking_result.trajectory_world = optimized_trajectory;
        } else {
            tracking_result.initialized = false;
            tracking_result.tracking_ok = false;
            tracking_result.trajectory_world = optimized_trajectory;
        }
        result.optimized_tracking_results.push_back(std::move(tracking_result));
    }

    return result;
}

}  // namespace newnewhand
