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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 0;
    int frames = -1;
    bool preview = true;
    bool glfw_view = true;
    bool verbose = true;
};

struct LatestTrackingData {
    newnewhand::StereoFrame stereo_frame;
    newnewhand::StereoCameraTrackingResult tracking_result;
    double tracking_fps = 0.0;
};

struct SharedWorkerState {
    std::mutex mutex;
    std::shared_ptr<const LatestTrackingData> latest_frame;
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
                << "Usage: stereo_camera_tracking_demo [options]\n"
                << "  --calibration <path>      default: " << DefaultCalibrationPath() << "\n"
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
    return options;
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

void DrawTrackingOverlay(
    cv::Mat& image,
    const std::string& label,
    const std::string& serial,
    const newnewhand::StereoCameraTrackingResult& tracking,
    double tracking_fps,
    double render_fps) {
    cv::putText(
        image,
        label + " serial: " + serial,
        cv::Point(16, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(40, 220, 255),
        2);

    std::ostringstream stats;
    stats
        << "slam "
        << (tracking.initialized ? (tracking.tracking_ok ? "ok" : "hold") : "init")
        << " track_fps=" << std::fixed << std::setprecision(1) << tracking_fps
        << " render_fps=" << std::fixed << std::setprecision(1) << render_fps
        << " kp=" << tracking.left_keypoints
        << " stereo=" << tracking.stereo_points
        << " match=" << tracking.matched_points
        << " inlier=" << tracking.tracking_inliers;
    cv::putText(
        image,
        stats.str(),
        cv::Point(16, 56),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(40, 220, 255),
        1);

    std::ostringstream disparity_stats;
    disparity_stats
        << "disp_kp=" << tracking.valid_disparity_keypoints
        << " nan=" << tracking.invalid_nonfinite_disparity
        << " low=" << tracking.invalid_low_disparity
        << " depth=" << tracking.invalid_depth;
    cv::putText(
        image,
        disparity_stats.str(),
        cv::Point(16, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(40, 220, 255),
        1);

    if (!tracking.status_message.empty()) {
        cv::putText(
            image,
            tracking.status_message,
            cv::Point(16, 104),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(40, 220, 255),
            1);
    }

    if (tracking.initialized) {
        std::ostringstream pose_stats;
        pose_stats
            << "xyz=("
            << tracking.camera_center_world[0] << ", "
            << tracking.camera_center_world[1] << ", "
            << tracking.camera_center_world[2] << ")";
        cv::putText(
            image,
            pose_stats.str(),
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
        const auto calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        const auto capture_config = BuildCaptureConfigFromCalibration(options, calibration);

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

                if (options.verbose) {
                    std::cerr << "[tracking] left serial=" << active_cameras[0].serial_number << "\n";
                    std::cerr << "[tracking] right serial=" << active_cameras[1].serial_number << "\n";
                }

                newnewhand::StereoVisualOdometryConfig tracking_config;
                tracking_config.calibration = calibration;
                tracking_config.verbose_logging = options.verbose;
                newnewhand::StereoVisualOdometry tracker(std::move(tracking_config));

                try {
                    double tracking_fps = 0.0;
                    auto previous_loop_time = std::chrono::steady_clock::now();
                    for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
                        if (stop_requested.load()) {
                            break;
                        }

                        const auto loop_start = std::chrono::steady_clock::now();
                        auto stereo_frame = capture.Capture();
                        if (stop_requested.load()) {
                            break;
                        }

                        auto tracking_frame = ToTrackingFrame(stereo_frame);
                        auto tracking_result = tracker.Track(tracking_frame);
                        const auto now = std::chrono::steady_clock::now();
                        const double dt_seconds =
                            std::chrono::duration<double>(now - previous_loop_time).count();
                        previous_loop_time = now;
                        if (dt_seconds > 1e-6) {
                            const double instant_fps = 1.0 / dt_seconds;
                            tracking_fps = tracking_fps <= 0.0
                                ? instant_fps
                                : 0.85 * tracking_fps + 0.15 * instant_fps;
                        }
                        if (options.verbose) {
                            std::cerr
                                << "[tracking] capture=" << tracking_result.capture_index
                                << " status=" << tracking_result.status_message
                                << " tracking_fps=" << tracking_fps
                                << " stereo=" << tracking_result.stereo_points
                                << " match=" << tracking_result.matched_points
                                << " inlier=" << tracking_result.tracking_inliers
                                << "\n";
                        }

                        auto latest_frame = std::make_shared<LatestTrackingData>();
                        latest_frame->stereo_frame = std::move(stereo_frame);
                        latest_frame->tracking_result = std::move(tracking_result);
                        latest_frame->tracking_fps = tracking_fps;
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

        std::shared_ptr<const LatestTrackingData> latest_frame;
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
                    << "newnewhand OpenGL Viewer"
                    << " | track "
                    << std::fixed << std::setprecision(1)
                    << (latest_frame ? latest_frame->tracking_fps : 0.0)
                    << " FPS"
                    << " | render "
                    << std::fixed << std::setprecision(1)
                    << render_fps
                    << " FPS";
                viewer.SetTitle(title.str());
                if (!viewer.Render(empty_fused_frame, tracking_result)) {
                    std::cerr << "[tracking] GLFW viewer closed by user\n";
                    stop_requested.store(true);
                    break;
                }
            }

            if (options.preview) {
                if (latest_frame) {
                    const std::array<std::string, 2> labels = {"LEFT", "RIGHT"};
                    const std::array<std::string, 2> serials = {
                        calibration.left_camera_serial_number,
                        calibration.right_camera_serial_number,
                    };
                    for (std::size_t i = 0; i < latest_frame->stereo_frame.views.size(); ++i) {
                        const auto& frame = latest_frame->stereo_frame.views[i];
                        if (!frame.valid || frame.bgr_image.empty()) {
                            continue;
                        }
                        cv::Mat preview;
                        cv::resize(frame.bgr_image, preview, cv::Size(), 0.5, 0.5);
                        DrawTrackingOverlay(
                            preview,
                            labels[i],
                            serials[i],
                            latest_frame->tracking_result,
                            latest_frame->tracking_fps,
                            render_fps);
                        cv::imshow("tracking_" + labels[i], preview);
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
