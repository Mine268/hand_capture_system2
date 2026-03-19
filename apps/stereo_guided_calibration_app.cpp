#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/capture/stereo_capture.h"

namespace {

struct AppOptions {
    std::filesystem::path output_path = "results/stereo_calibration.yaml";
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int capture_fps = 3;
    int target_frames = 30;
    bool use_find_chessboard_sb = true;
};

struct SelectedStereoPair {
    std::string left_serial;
    std::string right_serial;
};

std::string TimestampString() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return oss.str();
}

AppOptions ParseArgs(int argc, char** argv) {
    AppOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };

        if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--exposure_us") {
            options.exposure_us = std::stof(require_value(arg));
        } else if (arg == "--gain") {
            options.gain = std::stof(require_value(arg));
        } else if (arg == "--fps") {
            options.capture_fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        } else if (arg == "--frames") {
            options.target_frames = std::stoi(require_value(arg));
        } else if (arg == "--use_sb") {
            options.use_find_chessboard_sb = true;
        } else if (arg == "--no_use_sb") {
            options.use_find_chessboard_sb = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_guided_calibration_app [options]\n"
                << "  --output <path>      default: results/stereo_calibration.yaml\n"
                << "  --exposure_us <f>    default: 10000\n"
                << "  --gain <f>           default: -1 (auto)\n"
                << "  --fps <int>          default: 3\n"
                << "  --frames <int>       default: 30 valid images per calibration stage\n"
                << "  --use_sb             default: enabled\n"
                << "  --no_use_sb          use classic findChessboardCorners\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.capture_fps == 0) {
        throw std::runtime_error("--fps must be positive");
    }
    if (options.target_frames <= 0) {
        throw std::runtime_error("--frames must be positive");
    }

    return options;
}

int PromptInt(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("failed to read terminal input");
        }
        std::istringstream iss(line);
        int value = 0;
        if (iss >> value && value > 0) {
            return value;
        }
        std::cout << "Please enter a positive integer.\n";
    }
}

float PromptFloat(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("failed to read terminal input");
        }
        std::istringstream iss(line);
        float value = 0.0f;
        if (iss >> value && value > 0.0f) {
            return value;
        }
        std::cout << "Please enter a positive float.\n";
    }
}

void OverlaySelectionPreview(
    cv::Mat& image,
    const std::string& title,
    const std::string& serial,
    const std::string& instruction) {
    cv::putText(
        image,
        title,
        cv::Point(16, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.75,
        cv::Scalar(0, 255, 255),
        2);
    cv::putText(
        image,
        "serial: " + serial,
        cv::Point(16, 58),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 255),
        2);
    cv::putText(
        image,
        instruction,
        cv::Point(16, 88),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 255, 255),
        2);
}

void OverlayCapturePreview(
    cv::Mat& image,
    const std::string& stage,
    const std::string& label,
    const std::string& serial,
    int saved_frames,
    int target_frames,
    bool detected) {
    cv::putText(
        image,
        stage + " | " + label + " serial: " + serial,
        cv::Point(16, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);
    cv::putText(
        image,
        "saved " + std::to_string(saved_frames) + "/" + std::to_string(target_frames),
        cv::Point(16, 56),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 255),
        2);
    cv::putText(
        image,
        detected ? "checkerboard detected" : "show checkerboard clearly",
        cv::Point(16, 84),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255),
        2);
}

void SaveImage(
    const std::filesystem::path& root,
    std::uint64_t capture_index,
    const cv::Mat& image) {
    std::filesystem::create_directories(root);
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << ".bmp";
    cv::imwrite((root / name.str()).string(), image);
}

SelectedStereoPair SelectLeftRightCameras(const AppOptions& options) {
    const auto connected = newnewhand::StereoCapture::EnumerateConnectedCameras();
    std::cout << "Connected cameras: " << connected.size() << "\n";
    for (const auto& descriptor : connected) {
        std::cout
            << "  index=" << descriptor.device_index
            << " serial=" << descriptor.serial_number
            << " model=" << descriptor.model_name
            << "\n";
    }
    if (connected.size() != 2) {
        throw std::runtime_error("guided calibration currently requires exactly two connected cameras");
    }

    newnewhand::StereoCaptureConfig capture_config;
    capture_config.camera_settings.exposure_us = options.exposure_us;
    capture_config.camera_settings.gain = options.gain;
    newnewhand::StereoCapture capture(capture_config);
    capture.Initialize();
    capture.Start();

    const auto active = capture.ActiveCameras();
    std::cout << "Preview camera order:\n";
    std::cout << "  cam0 serial=" << active[0].serial_number << "\n";
    std::cout << "  cam1 serial=" << active[1].serial_number << "\n";
    std::cout << "Press 1 if cam0 is LEFT and cam1 is RIGHT.\n";
    std::cout << "Press 2 if cam1 is LEFT and cam0 is RIGHT.\n";
    std::cout << "Press q to quit.\n";

    SelectedStereoPair selected;
    while (true) {
        const auto stereo_frame = capture.Capture();

        for (std::size_t i = 0; i < stereo_frame.views.size(); ++i) {
            const auto& frame = stereo_frame.views[i];
            if (!frame.valid || frame.bgr_image.empty()) {
                continue;
            }
            cv::Mat preview;
            cv::resize(frame.bgr_image, preview, cv::Size(), 0.5, 0.5);
            OverlaySelectionPreview(
                preview,
                "preview cam" + std::to_string(i),
                active[i].serial_number,
                i == 0 ? "press 1 if this is LEFT" : "press 2 if this is LEFT");
            cv::imshow("select_cam" + std::to_string(i), preview);
        }

        const int key = cv::waitKey(1);
        if (key == '1') {
            selected.left_serial = active[0].serial_number;
            selected.right_serial = active[1].serial_number;
            break;
        }
        if (key == '2') {
            selected.left_serial = active[1].serial_number;
            selected.right_serial = active[0].serial_number;
            break;
        }
        if (key == 'q' || key == 27) {
            throw std::runtime_error("camera-role selection cancelled by user");
        }
    }

    capture.Stop();
    cv::destroyWindow("select_cam0");
    cv::destroyWindow("select_cam1");
    return selected;
}

newnewhand::CheckerboardConfig PromptCheckerboardConfig() {
    newnewhand::CheckerboardConfig checkerboard;
    checkerboard.inner_corners_cols = PromptInt("Enter checkerboard inner corners cols: ");
    checkerboard.inner_corners_rows = PromptInt("Enter checkerboard inner corners rows: ");
    checkerboard.square_size = PromptFloat("Enter checkerboard square size in meters: ");
    return checkerboard;
}

std::filesystem::path BuildCaptureRoot(const std::filesystem::path& output_path) {
    const std::filesystem::path base_dir =
        output_path.parent_path().empty() ? std::filesystem::path("results") : output_path.parent_path();
    return base_dir / (output_path.stem().string() + "_capture_" + TimestampString());
}

std::vector<std::filesystem::path> CollectImagePaths(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.is_regular_file()) {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

void EnsureActiveStereoOrder(
    newnewhand::StereoCapture& capture,
    const SelectedStereoPair& selected) {
    const auto active = capture.ActiveCameras();
    if (active[0].serial_number != selected.left_serial || active[1].serial_number != selected.right_serial) {
        capture.Stop();
        throw std::runtime_error("failed to enforce selected left/right serial order during capture");
    }
}

std::vector<std::filesystem::path> CaptureSingleCalibrationFrames(
    const AppOptions& options,
    const SelectedStereoPair& selected,
    const newnewhand::CheckerboardConfig& checkerboard,
    const std::string& stage_name,
    std::size_t target_view_index,
    const std::filesystem::path& output_dir) {
    newnewhand::StereoCalibrationConfig calibration_config;
    calibration_config.checkerboard = checkerboard;
    calibration_config.use_find_chessboard_sb = options.use_find_chessboard_sb;
    newnewhand::StereoCalibrator calibrator(calibration_config);

    newnewhand::StereoCaptureConfig capture_config;
    capture_config.serial_numbers = {selected.left_serial, selected.right_serial};
    capture_config.camera_settings.exposure_us = options.exposure_us;
    capture_config.camera_settings.gain = options.gain;
    newnewhand::StereoCapture capture(capture_config);
    capture.Initialize();
    capture.Start();
    EnsureActiveStereoOrder(capture, selected);

    std::cout << "Stage: " << stage_name << "\n";
    std::cout << "Capturing valid checkerboard images at " << options.capture_fps << " FPS.\n";
    std::cout << "Move the checkerboard through different positions and angles.\n";
    std::cout << "Will save " << options.target_frames << " valid images.\n";

    const auto frame_interval = std::chrono::microseconds(1000000 / options.capture_fps);
    int saved_frames = 0;
    while (saved_frames < options.target_frames) {
        const auto loop_start = std::chrono::steady_clock::now();
        const auto stereo_frame = capture.Capture();
        if (!stereo_frame.is_complete()) {
            continue;
        }

        const auto& frame = stereo_frame.views[target_view_index];
        std::vector<cv::Point2f> corners;
        const bool detected = calibrator.DetectSingleCorners(frame.bgr_image, corners);

        cv::Mat preview = frame.bgr_image.clone();
        if (detected) {
            const cv::Size board_size(
                checkerboard.inner_corners_cols,
                checkerboard.inner_corners_rows);
            cv::drawChessboardCorners(preview, board_size, corners, true);
        }
        OverlayCapturePreview(
            preview,
            stage_name,
            target_view_index == 0 ? "LEFT" : "RIGHT",
            frame.serial_number,
            saved_frames,
            options.target_frames,
            detected);

        cv::Mat preview_view;
        cv::resize(preview, preview_view, cv::Size(), 0.5, 0.5);
        cv::imshow("guided_calib_" + stage_name, preview_view);

        if (detected) {
            const std::uint64_t image_index = static_cast<std::uint64_t>(saved_frames + 1);
            SaveImage(output_dir, image_index, frame.bgr_image);
            saved_frames += 1;
            std::cout << "Saved valid image " << saved_frames << "/" << options.target_frames << "\n";
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            capture.Stop();
            throw std::runtime_error("guided calibration capture cancelled by user");
        }

        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        const auto remaining = frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        if (remaining.count() > 0) {
            std::this_thread::sleep_for(remaining);
        }
    }

    capture.Stop();
    cv::destroyWindow("guided_calib_" + stage_name);
    return CollectImagePaths(output_dir);
}

std::vector<newnewhand::CalibrationImagePair> CaptureStereoCalibrationPairs(
    const AppOptions& options,
    const SelectedStereoPair& selected,
    const newnewhand::CheckerboardConfig& checkerboard,
    const std::filesystem::path& left_dir,
    const std::filesystem::path& right_dir) {
    newnewhand::StereoCalibrationConfig calibration_config;
    calibration_config.checkerboard = checkerboard;
    calibration_config.use_find_chessboard_sb = options.use_find_chessboard_sb;
    newnewhand::StereoCalibrator calibrator(calibration_config);

    newnewhand::StereoCaptureConfig capture_config;
    capture_config.serial_numbers = {selected.left_serial, selected.right_serial};
    capture_config.camera_settings.exposure_us = options.exposure_us;
    capture_config.camera_settings.gain = options.gain;
    newnewhand::StereoCapture capture(capture_config);
    capture.Initialize();
    capture.Start();
    EnsureActiveStereoOrder(capture, selected);

    std::cout << "Stage: STEREO_EXTRINSICS\n";
    std::cout << "Capturing valid stereo checkerboard pairs at " << options.capture_fps << " FPS.\n";
    std::cout << "Will save " << options.target_frames << " valid stereo pairs.\n";

    const auto frame_interval = std::chrono::microseconds(1000000 / options.capture_fps);
    int saved_pairs = 0;
    while (saved_pairs < options.target_frames) {
        const auto loop_start = std::chrono::steady_clock::now();
        const auto stereo_frame = capture.Capture();
        if (!stereo_frame.is_complete()) {
            continue;
        }

        std::vector<cv::Point2f> left_corners;
        std::vector<cv::Point2f> right_corners;
        const bool detected = calibrator.DetectStereoCorners(
            stereo_frame.views[0].bgr_image,
            stereo_frame.views[1].bgr_image,
            left_corners,
            right_corners);

        cv::Mat left_preview = stereo_frame.views[0].bgr_image.clone();
        cv::Mat right_preview = stereo_frame.views[1].bgr_image.clone();
        if (detected) {
            const cv::Size board_size(
                checkerboard.inner_corners_cols,
                checkerboard.inner_corners_rows);
            cv::drawChessboardCorners(left_preview, board_size, left_corners, true);
            cv::drawChessboardCorners(right_preview, board_size, right_corners, true);
        }
        OverlayCapturePreview(
            left_preview,
            "STEREO_EXTRINSICS",
            "LEFT",
            selected.left_serial,
            saved_pairs,
            options.target_frames,
            detected);
        OverlayCapturePreview(
            right_preview,
            "STEREO_EXTRINSICS",
            "RIGHT",
            selected.right_serial,
            saved_pairs,
            options.target_frames,
            detected);

        cv::Mat left_view;
        cv::Mat right_view;
        cv::resize(left_preview, left_view, cv::Size(), 0.5, 0.5);
        cv::resize(right_preview, right_view, cv::Size(), 0.5, 0.5);
        cv::imshow("guided_calib_stereo_left", left_view);
        cv::imshow("guided_calib_stereo_right", right_view);

        if (detected) {
            const std::uint64_t image_index = static_cast<std::uint64_t>(saved_pairs + 1);
            SaveImage(left_dir, image_index, stereo_frame.views[0].bgr_image);
            SaveImage(right_dir, image_index, stereo_frame.views[1].bgr_image);
            saved_pairs += 1;
            std::cout << "Saved valid stereo pair " << saved_pairs << "/" << options.target_frames << "\n";
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            capture.Stop();
            throw std::runtime_error("guided stereo capture cancelled by user");
        }

        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        const auto remaining = frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        if (remaining.count() > 0) {
            std::this_thread::sleep_for(remaining);
        }
    }

    capture.Stop();
    cv::destroyWindow("guided_calib_stereo_left");
    cv::destroyWindow("guided_calib_stereo_right");
    return newnewhand::StereoCalibrator::CollectImagePairs(left_dir, right_dir);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppOptions options = ParseArgs(argc, argv);
        const SelectedStereoPair selected = SelectLeftRightCameras(options);

        std::cout << "Selected stereo order:\n";
        std::cout << "  left_serial=" << selected.left_serial << "\n";
        std::cout << "  right_serial=" << selected.right_serial << "\n";

        const newnewhand::CheckerboardConfig checkerboard = PromptCheckerboardConfig();
        const std::filesystem::path capture_root = BuildCaptureRoot(options.output_path);
        const std::filesystem::path mono_left_dir = capture_root / "mono_left";
        const std::filesystem::path mono_right_dir = capture_root / "mono_right";
        const std::filesystem::path stereo_left_dir = capture_root / "stereo_left";
        const std::filesystem::path stereo_right_dir = capture_root / "stereo_right";

        const auto mono_left_images = CaptureSingleCalibrationFrames(
            options,
            selected,
            checkerboard,
            "LEFT_MONO",
            0,
            mono_left_dir);
        const auto mono_right_images = CaptureSingleCalibrationFrames(
            options,
            selected,
            checkerboard,
            "RIGHT_MONO",
            1,
            mono_right_dir);
        const auto stereo_pairs = CaptureStereoCalibrationPairs(
            options,
            selected,
            checkerboard,
            stereo_left_dir,
            stereo_right_dir);

        newnewhand::StereoCalibrationConfig calibration_config;
        calibration_config.checkerboard = checkerboard;
        calibration_config.use_find_chessboard_sb = options.use_find_chessboard_sb;
        newnewhand::StereoCalibrator calibrator(calibration_config);

        const auto left_calibration = calibrator.CalibrateSingle(mono_left_images);
        const auto right_calibration = calibrator.CalibrateSingle(mono_right_images);
        auto result = calibrator.CalibrateStereoExtrinsics(
            stereo_pairs,
            left_calibration,
            right_calibration);
        result.left_camera_serial_number = selected.left_serial;
        result.right_camera_serial_number = selected.right_serial;
        calibrator.SaveResult(result, options.output_path);

        std::cout << "Calibration completed.\n";
        std::cout << "Captured pairs directory: " << capture_root << "\n";
        std::cout << "Left monocular valid images: " << left_calibration.observations.size() << "\n";
        std::cout << "Right monocular valid images: " << right_calibration.observations.size() << "\n";
        std::cout << "Valid calibration pairs: " << result.observations.size() << "\n";
        std::cout << "Image size: " << result.image_size.width << "x" << result.image_size.height << "\n";
        std::cout << "Left RMS: " << result.left_rms << "\n";
        std::cout << "Right RMS: " << result.right_rms << "\n";
        std::cout << "Stereo RMS: " << result.stereo_rms << "\n";
        std::cout << "Saved calibration: " << options.output_path << "\n";
        std::cout << "Saved serials: left=" << result.left_camera_serial_number
                  << " right=" << result.right_camera_serial_number << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
