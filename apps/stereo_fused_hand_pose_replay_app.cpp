#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/render/glfw_scene_viewer.h"

namespace {

struct DemoOptions {
    std::filesystem::path input_dir;
    unsigned int playback_fps = 30;
    int frames = -1;
    bool preview = true;
    bool glfw_view = true;
    bool verbose = true;
};

struct ReplayFrame {
    std::uint64_t capture_index = 0;
    std::array<cv::Mat, 2> images;
    std::array<bool, 2> has_image = {false, false};
    newnewhand::StereoFusedHandPoseFrame fused_frame;
    std::optional<newnewhand::StereoCameraTrackingResult> tracking_result;
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
        else if (arg == "--fps") options.playback_fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--glfw_view") options.glfw_view = true;
        else if (arg == "--no_glfw_view") options.glfw_view = false;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--quiet") options.verbose = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_fused_hand_pose_replay_app [options]\n"
                << "  --input_dir <dir>         required, saved replay package root\n"
                << "  --fps <int>               default: 30\n"
                << "  --frames <int>            default: -1 (use all frames)\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
                << "  --verbose | --quiet       default: --verbose\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.input_dir.empty()) {
        throw std::runtime_error("--input_dir is required");
    }
    if (options.playback_fps == 0) {
        throw std::runtime_error("--fps must be positive");
    }
    return options;
}

std::vector<std::filesystem::path> CollectFrameYamlPaths(const std::filesystem::path& input_dir) {
    const std::filesystem::path frames_dir = input_dir / "frames";
    if (!std::filesystem::is_directory(frames_dir)) {
        throw std::runtime_error("input_dir does not contain frames/: " + frames_dir.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(frames_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".yaml") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool ReadBoolNode(const cv::FileNode& node, bool default_value = false) {
    if (node.empty()) {
        return default_value;
    }
    int value = default_value ? 1 : 0;
    node >> value;
    return value != 0;
}

int ReadIntNode(const cv::FileNode& node, int default_value = 0) {
    if (node.empty()) {
        return default_value;
    }
    int value = default_value;
    node >> value;
    return value;
}

float ReadFloatNode(const cv::FileNode& node, float default_value = 0.0f) {
    if (node.empty()) {
        return default_value;
    }
    float value = default_value;
    node >> value;
    return value;
}

std::string ReadStringNode(const cv::FileNode& node) {
    if (node.empty()) {
        return {};
    }
    std::string value;
    node >> value;
    return value;
}

cv::Mat ReadFloatMat(const cv::FileNode& node) {
    cv::Mat value;
    if (!node.empty()) {
        node >> value;
        if (!value.empty() && value.type() != CV_32F) {
            value.convertTo(value, CV_32F);
        }
    }
    return value;
}

cv::Vec3f ReadVec3f(const cv::FileNode& node) {
    const cv::Mat value = ReadFloatMat(node);
    if (value.empty()) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }

    cv::Mat reshaped = value.reshape(1, 3);
    return cv::Vec3f(
        reshaped.at<float>(0, 0),
        reshaped.at<float>(1, 0),
        reshaped.at<float>(2, 0));
}

cv::Matx33f ReadMatx33f(const cv::FileNode& node) {
    const cv::Mat value = ReadFloatMat(node);
    if (value.empty()) {
        return cv::Matx33f::eye();
    }
    if (value.rows != 3 || value.cols != 3) {
        throw std::runtime_error("tracking rotation matrix must be 3x3");
    }

    cv::Matx33f output = cv::Matx33f::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            output(r, c) = value.at<float>(r, c);
        }
    }
    return output;
}

template <std::size_t Count>
void ReadFloatVector(const cv::FileNode& node, float (&dst)[Count]) {
    const cv::Mat value = ReadFloatMat(node);
    if (value.empty()) {
        return;
    }
    if (static_cast<std::size_t>(value.total()) != Count) {
        throw std::runtime_error("unexpected vector size in replay yaml");
    }

    const cv::Mat reshaped = value.reshape(1, static_cast<int>(Count));
    for (std::size_t i = 0; i < Count; ++i) {
        dst[i] = reshaped.at<float>(static_cast<int>(i), 0);
    }
}

template <std::size_t Rows, std::size_t Cols>
void ReadFloatMatrix(const cv::FileNode& node, float (&dst)[Rows][Cols]) {
    const cv::Mat value = ReadFloatMat(node);
    if (value.empty()) {
        return;
    }
    if (value.rows != static_cast<int>(Rows) || value.cols != static_cast<int>(Cols)) {
        throw std::runtime_error("unexpected matrix size in replay yaml");
    }

    for (std::size_t r = 0; r < Rows; ++r) {
        for (std::size_t c = 0; c < Cols; ++c) {
            dst[r][c] = value.at<float>(static_cast<int>(r), static_cast<int>(c));
        }
    }
}

newnewhand::HandPoseResult ReadPoseCam0(const cv::FileNode& node, bool is_right) {
    newnewhand::HandPoseResult pose;
    pose.detection.is_right = is_right;
    pose.detection.confidence = ReadFloatNode(node["confidence"]);
    ReadFloatVector(node["bbox"], pose.detection.bbox);
    ReadFloatVector(node["pred_cam"], pose.pred_cam);
    ReadFloatVector(node["camera_translation"], pose.camera_translation);
    ReadFloatVector(node["global_orient"], pose.global_orient);
    ReadFloatVector(node["betas"], pose.betas);
    ReadFloatMatrix(node["hand_pose"], pose.hand_pose);
    ReadFloatMatrix(node["keypoints_2d"], pose.keypoints_2d);
    ReadFloatMatrix(node["keypoints_3d_cam0"], pose.keypoints_3d);
    ReadFloatMatrix(node["vertices_cam0"], pose.vertices);
    return pose;
}

newnewhand::FusedHandPose ReadFusedHand(const cv::FileNode& node) {
    newnewhand::FusedHandPose hand;
    hand.is_right = ReadBoolNode(node["is_right"], true);
    hand.fused_from_stereo = ReadBoolNode(node["fused_from_stereo"], false);
    hand.has_view0 = ReadBoolNode(node["has_view0"], false);
    hand.has_view1 = ReadBoolNode(node["has_view1"], false);
    hand.root_joint_cam0 = ReadVec3f(node["root_joint_cam0"]);
    hand.pose_cam0 = ReadPoseCam0(node["pose_cam0"], hand.is_right);
    return hand;
}

std::vector<cv::Vec3f> ReadTrajectory(const cv::FileNode& node) {
    const cv::Mat value = ReadFloatMat(node);
    if (value.empty()) {
        return {};
    }
    if (value.cols != 3) {
        throw std::runtime_error("tracking trajectory must be Nx3");
    }

    std::vector<cv::Vec3f> trajectory;
    trajectory.reserve(static_cast<std::size_t>(value.rows));
    for (int r = 0; r < value.rows; ++r) {
        trajectory.emplace_back(
            value.at<float>(r, 0),
            value.at<float>(r, 1),
            value.at<float>(r, 2));
    }
    return trajectory;
}

newnewhand::StereoCameraTrackingResult ReadTrackingResult(const cv::FileNode& node) {
    newnewhand::StereoCameraTrackingResult tracking;
    tracking.capture_index = static_cast<std::uint64_t>(ReadIntNode(node["capture_index"]));
    tracking.initialized = ReadBoolNode(node["initialized"], false);
    tracking.tracking_ok = ReadBoolNode(node["tracking_ok"], false);
    tracking.reinitialized = ReadBoolNode(node["reinitialized"], false);
    tracking.left_keypoints = ReadIntNode(node["left_keypoints"]);
    tracking.stereo_points = ReadIntNode(node["stereo_points"]);
    tracking.valid_disparity_keypoints = ReadIntNode(node["valid_disparity_keypoints"]);
    tracking.invalid_nonfinite_disparity = ReadIntNode(node["invalid_nonfinite_disparity"]);
    tracking.invalid_low_disparity = ReadIntNode(node["invalid_low_disparity"]);
    tracking.invalid_depth = ReadIntNode(node["invalid_depth"]);
    tracking.valid_disparity_pixels = ReadIntNode(node["valid_disparity_pixels"]);
    tracking.min_valid_disparity = ReadFloatNode(node["min_valid_disparity"]);
    tracking.max_valid_disparity = ReadFloatNode(node["max_valid_disparity"]);
    tracking.matched_points = ReadIntNode(node["matched_points"]);
    tracking.tracking_inliers = ReadIntNode(node["tracking_inliers"]);
    tracking.rotation_world_from_cam0 = ReadMatx33f(node["rotation_world_from_cam0"]);
    tracking.camera_center_world = ReadVec3f(node["camera_center_world"]);
    tracking.status_message = ReadStringNode(node["status_message"]);
    tracking.trajectory_world = ReadTrajectory(node["trajectory_world"]);
    return tracking;
}

cv::Mat LoadImageRelativeToYaml(const std::filesystem::path& yaml_path, const cv::FileNode& path_node) {
    const std::string relative_path = ReadStringNode(path_node);
    if (relative_path.empty()) {
        return {};
    }
    return cv::imread((yaml_path.parent_path() / relative_path).string(), cv::IMREAD_COLOR);
}

ReplayFrame LoadReplayFrame(const std::filesystem::path& yaml_path) {
    cv::FileStorage fs(yaml_path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open replay yaml: " + yaml_path.string());
    }

    ReplayFrame frame;
    {
        double capture_index_value = 0.0;
        fs["capture_index"] >> capture_index_value;
        frame.capture_index = static_cast<std::uint64_t>(capture_index_value);
    }
    frame.fused_frame.capture_index = frame.capture_index;

    const cv::FileNode views = fs["views"];
    for (const auto& view_node : views) {
        const int camera_index = ReadIntNode(view_node["camera_index"], -1);
        if (camera_index < 0 || camera_index >= 2) {
            continue;
        }
        const std::size_t idx = static_cast<std::size_t>(camera_index);
        cv::Mat image = LoadImageRelativeToYaml(yaml_path, view_node["overlay_path"]);
        if (image.empty()) {
            image = LoadImageRelativeToYaml(yaml_path, view_node["image_path"]);
        }
        frame.has_image[idx] = !image.empty();
        frame.images[idx] = std::move(image);
    }

    const cv::FileNode fused_hands = fs["fused_hands"];
    for (const auto& hand_node : fused_hands) {
        frame.fused_frame.hands.push_back(ReadFusedHand(hand_node));
    }

    const cv::FileNode tracking_node = fs["tracking"];
    if (!tracking_node.empty()) {
        frame.tracking_result = ReadTrackingResult(tracking_node);
        frame.tracking_result->capture_index = frame.capture_index;
    }

    return frame;
}

void DrawReplayOverlay(
    cv::Mat& image,
    const std::string& label,
    const ReplayFrame& frame,
    double replay_fps) {
    cv::putText(
        image,
        label,
        cv::Point(16, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 255),
        2,
        cv::LINE_AA);

    std::ostringstream summary;
    summary << "capture=" << frame.capture_index
            << " hands=" << frame.fused_frame.hands.size()
            << " fps=" << std::fixed << std::setprecision(1) << replay_fps;
    cv::putText(
        image,
        summary.str(),
        cv::Point(16, 64),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(40, 220, 255),
        2,
        cv::LINE_AA);

    if (!frame.tracking_result.has_value()) {
        return;
    }

    const auto& tracking = *frame.tracking_result;
    std::ostringstream tracking_summary;
    tracking_summary
        << "tracking=" << (tracking.initialized ? (tracking.tracking_ok ? "ok" : "hold") : "init")
        << " match=" << tracking.matched_points
        << " inlier=" << tracking.tracking_inliers;
    cv::putText(
        image,
        tracking_summary.str(),
        cv::Point(16, 96),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(40, 220, 255),
        2,
        cv::LINE_AA);

    if (!tracking.status_message.empty()) {
        cv::putText(
            image,
            tracking.status_message,
            cv::Point(16, 128),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(40, 220, 255),
            1,
            cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);
        const auto calibration = newnewhand::StereoCalibrator::LoadResult(
            (options.input_dir / "calibration" / "stereo_calibration.yaml").string());
        auto frame_paths = CollectFrameYamlPaths(options.input_dir);
        if (frame_paths.empty()) {
            throw std::runtime_error("no replay frame yaml files found");
        }
        if (options.frames > 0 && static_cast<std::size_t>(options.frames) < frame_paths.size()) {
            frame_paths.resize(static_cast<std::size_t>(options.frames));
        }

        newnewhand::GlfwSceneViewerConfig viewer_config;
        viewer_config.draw_world_axes = true;
        viewer_config.has_cam1_pose = true;
        viewer_config.cam1_rotation_cam1_to_cam0 = cv::Matx33f(calibration.rotation).t();
        const cv::Vec3f translation_01(calibration.translation);
        viewer_config.cam1_center_cam0 = -(viewer_config.cam1_rotation_cam1_to_cam0 * translation_01);
        newnewhand::GlfwSceneViewer viewer(viewer_config);
        if (options.glfw_view && !viewer.Initialize()) {
            throw std::runtime_error("failed to initialize GLFW replay viewer");
        }

        const auto playback_interval = std::chrono::microseconds(1000000 / options.playback_fps);
        double replay_fps = 0.0;
        auto previous_loop_time = std::chrono::steady_clock::now();

        for (const auto& frame_path : frame_paths) {
            const auto loop_start = std::chrono::steady_clock::now();
            ReplayFrame frame = LoadReplayFrame(frame_path);

            const auto now = std::chrono::steady_clock::now();
            const double dt_seconds = std::chrono::duration<double>(now - previous_loop_time).count();
            previous_loop_time = now;
            if (dt_seconds > 1e-6) {
                const double instant_fps = 1.0 / dt_seconds;
                replay_fps = replay_fps <= 0.0
                    ? instant_fps
                    : 0.85 * replay_fps + 0.15 * instant_fps;
            }

            if (options.verbose) {
                std::cerr
                    << "[replay] capture=" << frame.capture_index
                    << " hands=" << frame.fused_frame.hands.size()
                    << " tracking="
                    << (frame.tracking_result.has_value() ? frame.tracking_result->status_message : "<none>")
                    << "\n";
            }

            if (options.glfw_view) {
                const newnewhand::StereoCameraTrackingResult* tracking =
                    frame.tracking_result.has_value() ? &*frame.tracking_result : nullptr;
                if (!viewer.Render(frame.fused_frame, tracking)) {
                    break;
                }
                viewer.SetTitle("Replay capture " + std::to_string(frame.capture_index));
            }

            if (options.preview) {
                const std::array<std::string, 2> labels = {"LEFT", "RIGHT"};
                for (std::size_t i = 0; i < frame.images.size(); ++i) {
                    if (!frame.has_image[i] || frame.images[i].empty()) {
                        continue;
                    }
                    cv::Mat preview;
                    cv::resize(frame.images[i], preview, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                    DrawReplayOverlay(preview, labels[i], frame, replay_fps);
                    cv::imshow("offline_replay_" + labels[i], preview);
                }
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    break;
                }
            }

            const auto elapsed = std::chrono::steady_clock::now() - loop_start;
            const auto remaining = playback_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
            if (remaining.count() > 0) {
                std::this_thread::sleep_for(remaining);
            }
        }

        if (options.preview) {
            cv::destroyAllWindows();
        }
        viewer.Shutdown();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
