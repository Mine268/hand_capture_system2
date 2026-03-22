#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/capture/stereo_capture.h"

namespace {

struct DemoOptions {
    std::string cam0_serial;
    std::string cam1_serial;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    int frames = 300;
    bool enable_gamma = false;
    bool metadata_only = false;
    bool preview = false;
    bool verbose = false;
    newnewhand::StereoPixelFormatMode pixel_format = newnewhand::StereoPixelFormatMode::kRawPreferred;
};

newnewhand::StereoPixelFormatMode ParsePixelFormatMode(const std::string& value) {
    if (value == "raw") {
        return newnewhand::StereoPixelFormatMode::kRawPreferred;
    }
    if (value == "mono8") {
        return newnewhand::StereoPixelFormatMode::kMono8;
    }
    if (value == "bgr8") {
        return newnewhand::StereoPixelFormatMode::kBgr8;
    }
    if (value == "rgb8") {
        return newnewhand::StereoPixelFormatMode::kRgb8;
    }

    throw std::runtime_error("unsupported --pixel_format: " + value);
}

const char* PixelFormatModeName(newnewhand::StereoPixelFormatMode mode) {
    switch (mode) {
        case newnewhand::StereoPixelFormatMode::kRawPreferred:
            return "raw";
        case newnewhand::StereoPixelFormatMode::kMono8:
            return "mono8";
        case newnewhand::StereoPixelFormatMode::kBgr8:
            return "bgr8";
        case newnewhand::StereoPixelFormatMode::kRgb8:
            return "rgb8";
    }

    return "unknown";
}

bool IsValidStereoFrame(const newnewhand::StereoFrame& stereo_frame) {
    return stereo_frame.views[0].valid && stereo_frame.views[1].valid;
}

cv::Mat MakeErrorTile(const std::string& label, const std::string& error_message, const cv::Size& size) {
    cv::Mat tile(size, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::putText(
        tile,
        label,
        cv::Point(24, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(0, 200, 255),
        2,
        cv::LINE_AA);
    cv::putText(
        tile,
        "capture failed",
        cv::Point(24, 90),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(0, 80, 255),
        2,
        cv::LINE_AA);
    cv::putText(
        tile,
        error_message.substr(0, 96),
        cv::Point(24, 140),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(220, 220, 220),
        1,
        cv::LINE_AA);
    return tile;
}

cv::Mat RenderCameraTile(
    const newnewhand::CameraFrame& frame,
    const newnewhand::StereoCaptureTimingCameraInfo& timing,
    const std::string& label,
    double ema_fps) {
    constexpr int kTileWidth = 960;
    constexpr int kTileHeight = 540;
    const cv::Size tile_size(kTileWidth, kTileHeight);

    if (!frame.valid || frame.bgr_image.empty()) {
        return MakeErrorTile(label, frame.error_message.empty() ? "no image data" : frame.error_message, tile_size);
    }

    cv::Mat tile;
    cv::resize(frame.bgr_image, tile, tile_size, 0.0, 0.0, cv::INTER_AREA);
    cv::putText(
        tile,
        label + " " + timing.pixel_format,
        cv::Point(18, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.85,
        cv::Scalar(0, 255, 255),
        2,
        cv::LINE_AA);

    std::ostringstream line1;
    line1 << "fps=" << std::fixed << std::setprecision(2) << ema_fps
          << " frame=" << frame.frame_index
          << " bytes=" << timing.frame_bytes;
    cv::putText(
        tile,
        line1.str(),
        cv::Point(18, 66),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(32, 240, 32),
        2,
        cv::LINE_AA);

    std::ostringstream line2;
    line2 << "wait_ms=" << std::fixed << std::setprecision(3) << timing.wait_frame_ms
          << " img_ms=" << timing.image_process_ms
          << " worker_ms=" << timing.total_worker_ms;
    cv::putText(
        tile,
        line2.str(),
        cv::Point(18, 98),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(32, 240, 32),
        2,
        cv::LINE_AA);

    return tile;
}

cv::Mat RenderStereoPreview(
    const newnewhand::StereoFrame& stereo_frame,
    const newnewhand::StereoCaptureTimingInfo& timing,
    double ema_fps,
    double capture_ms) {
    cv::Mat cam0 = RenderCameraTile(stereo_frame.views[0], timing.cameras[0], "cam0", ema_fps);
    cv::Mat cam1 = RenderCameraTile(stereo_frame.views[1], timing.cameras[1], "cam1", ema_fps);

    cv::Mat preview;
    cv::hconcat(std::vector<cv::Mat>{cam0, cam1}, preview);

    std::ostringstream header;
    header << "capture=" << stereo_frame.capture_index
           << " capture_ms=" << std::fixed << std::setprecision(3) << capture_ms
           << " trigger_ms=" << timing.trigger_ms
           << " assemble_ms=" << timing.assemble_ms;
    cv::putText(
        preview,
        header.str(),
        cv::Point(20, preview.rows - 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);

    return preview;
}

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

        if (arg == "--cam0_serial") options.cam0_serial = require_value(arg);
        else if (arg == "--cam1_serial") options.cam1_serial = require_value(arg);
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--pixel_format") options.pixel_format = ParsePixelFormatMode(require_value(arg));
        else if (arg == "--gamma") options.enable_gamma = true;
        else if (arg == "--metadata_only") options.metadata_only = true;
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--quiet") options.verbose = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_capture_benchmark_app [options]\n"
                << "  --cam0_serial <serial>    default: auto select\n"
                << "  --cam1_serial <serial>    default: auto select\n"
                << "  --exposure_us <float>     default: 10000\n"
                << "  --gain <float>            default: -1 (auto)\n"
                << "  --frames <int>            default: 300\n"
                << "  --pixel_format <mode>     raw | mono8 | bgr8 | rgb8, default: raw\n"
                << "  --gamma                   enable camera gamma, default: off\n"
                << "  --metadata_only           skip image conversion/copy, only time metadata path\n"
                << "  --preview | --no_preview  default: --no_preview\n"
                << "  --verbose | --quiet       default: --quiet\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.frames <= 0) {
        throw std::runtime_error("--frames must be positive");
    }
    if (options.preview && options.metadata_only) {
        throw std::runtime_error("--preview cannot be used together with --metadata_only");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);

        newnewhand::StereoCaptureConfig config;
        config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        config.camera_settings.exposure_us = options.exposure_us;
        config.camera_settings.gain = options.gain;
        config.camera_settings.enable_gamma = options.enable_gamma;
        config.camera_settings.include_image_data = !options.metadata_only;
        config.camera_settings.pixel_format = options.pixel_format;

        newnewhand::StereoCapture capture(config);
        capture.Initialize();
        capture.Start();
        const auto active = capture.ActiveCameras();
        const auto runtime = capture.RuntimeInfo();

        std::cout << "Stereo benchmark order:\n";
        std::cout << "  cam0 serial=" << active[0].serial_number << " model=" << active[0].model_name << "\n";
        std::cout << "  cam1 serial=" << active[1].serial_number << " model=" << active[1].model_name << "\n";
        std::cout << "Stereo benchmark config:\n";
        std::cout << "  requested_pixel_format=" << PixelFormatModeName(options.pixel_format) << "\n";
        std::cout << "  gamma=" << (options.enable_gamma ? "on" : "off") << "\n";
        std::cout << "  include_image_data=" << (options.metadata_only ? "false" : "true") << "\n";
        std::cout << "  preview=" << (options.preview ? "true" : "false") << "\n";
        std::cout << "Stereo runtime info:\n";
        std::cout
            << "  cam0 pixel_format=" << runtime[0].pixel_format
            << " payload_size=" << runtime[0].payload_size << "\n";
        std::cout
            << "  cam1 pixel_format=" << runtime[1].pixel_format
            << " payload_size=" << runtime[1].payload_size << "\n";

        const auto benchmark_start = std::chrono::steady_clock::now();
        auto previous_trigger_time = std::chrono::steady_clock::time_point{};
        double ema_fps = 0.0;
        double total_capture_ms = 0.0;
        double total_sdk_capture_ms = 0.0;
        double total_trigger_ms = 0.0;
        double total_assemble_ms = 0.0;
        std::array<double, 2> total_wait_frame_ms = {0.0, 0.0};
        std::array<double, 2> total_image_process_ms = {0.0, 0.0};
        std::array<double, 2> total_worker_ms = {0.0, 0.0};
        std::array<std::uint64_t, 2> total_frame_bytes = {0, 0};
        int processed_frames = 0;
        int valid_stereo_frames = 0;

        for (int frame_index = 0; frame_index < options.frames; ++frame_index) {
            const auto capture_start = std::chrono::steady_clock::now();
            const auto stereo_frame = capture.Capture();
            const auto capture_end = std::chrono::steady_clock::now();
            const auto timing = capture.LastCaptureTiming();

            const double capture_ms =
                std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
            total_capture_ms += capture_ms;
            total_sdk_capture_ms += timing.total_capture_ms;
            total_trigger_ms += timing.trigger_ms;
            total_assemble_ms += timing.assemble_ms;
            for (std::size_t camera_index = 0; camera_index < timing.cameras.size(); ++camera_index) {
                total_wait_frame_ms[camera_index] += timing.cameras[camera_index].wait_frame_ms;
                total_image_process_ms[camera_index] += timing.cameras[camera_index].image_process_ms;
                total_worker_ms[camera_index] += timing.cameras[camera_index].total_worker_ms;
                total_frame_bytes[camera_index] += timing.cameras[camera_index].frame_bytes;
            }

            if (previous_trigger_time != std::chrono::steady_clock::time_point{}) {
                const double dt_seconds =
                    std::chrono::duration<double>(stereo_frame.trigger_timestamp - previous_trigger_time).count();
                if (dt_seconds > 1e-6) {
                    const double instant_fps = 1.0 / dt_seconds;
                    ema_fps = ema_fps <= 0.0
                        ? instant_fps
                        : 0.85 * ema_fps + 0.15 * instant_fps;
                }
            }
            previous_trigger_time = stereo_frame.trigger_timestamp;

            const bool valid_stereo = IsValidStereoFrame(stereo_frame);
            if (valid_stereo) {
                valid_stereo_frames += 1;
            }
            processed_frames += 1;

            if (options.verbose) {
                std::cout
                    << "capture=" << stereo_frame.capture_index
                    << " valid=" << (valid_stereo ? 1 : 0)
                    << " capture_ms=" << std::fixed << std::setprecision(3) << capture_ms
                    << " sdk_ms=" << timing.total_capture_ms
                    << " trigger_ms=" << timing.trigger_ms
                    << " assemble_ms=" << timing.assemble_ms
                    << " cam0_fmt=" << timing.cameras[0].pixel_format
                    << " cam0_wait_ms=" << timing.cameras[0].wait_frame_ms
                    << " cam0_img_ms=" << timing.cameras[0].image_process_ms
                    << " cam0_bytes=" << timing.cameras[0].frame_bytes
                    << " cam1_fmt=" << timing.cameras[1].pixel_format
                    << " cam1_wait_ms=" << timing.cameras[1].wait_frame_ms
                    << " cam1_img_ms=" << timing.cameras[1].image_process_ms
                    << " cam1_bytes=" << timing.cameras[1].frame_bytes
                    << " fps=" << std::fixed << std::setprecision(2) << ema_fps
                    << "\n";
            }

            if (options.preview) {
                cv::Mat preview = RenderStereoPreview(stereo_frame, timing, ema_fps, capture_ms);
                cv::imshow("stereo_capture_benchmark", preview);
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    break;
                }
            }
        }

        const auto benchmark_end = std::chrono::steady_clock::now();
        const double total_seconds =
            std::chrono::duration<double>(benchmark_end - benchmark_start).count();
        const int completed_frames = std::max(1, processed_frames);
        const double average_fps = processed_frames / std::max(1e-6, total_seconds);
        const double average_capture_ms = total_capture_ms / static_cast<double>(completed_frames);
        const double average_sdk_capture_ms = total_sdk_capture_ms / static_cast<double>(completed_frames);

        capture.Stop();
        if (options.preview) {
            cv::destroyAllWindows();
        }

        std::cout << "Stereo capture benchmark summary:\n";
        std::cout << "  requested_frames=" << options.frames << "\n";
        std::cout << "  processed_frames=" << processed_frames << "\n";
        std::cout << "  valid_stereo_frames=" << valid_stereo_frames << "\n";
        std::cout << "  total_seconds=" << std::fixed << std::setprecision(3) << total_seconds << "\n";
        std::cout << "  average_fps=" << std::fixed << std::setprecision(2) << average_fps << "\n";
        std::cout << "  ema_fps_last=" << std::fixed << std::setprecision(2) << ema_fps << "\n";
        std::cout << "  average_capture_ms=" << std::fixed << std::setprecision(3) << average_capture_ms << "\n";
        std::cout << "  average_sdk_capture_ms=" << std::fixed << std::setprecision(3) << average_sdk_capture_ms << "\n";
        std::cout << "  average_trigger_ms=" << std::fixed << std::setprecision(3)
                  << (total_trigger_ms / static_cast<double>(completed_frames)) << "\n";
        std::cout << "  average_assemble_ms=" << std::fixed << std::setprecision(3)
                  << (total_assemble_ms / static_cast<double>(completed_frames)) << "\n";
        for (std::size_t camera_index = 0; camera_index < 2; ++camera_index) {
            std::cout
                << "  cam" << camera_index
                << "_average_wait_frame_ms=" << std::fixed << std::setprecision(3)
                << (total_wait_frame_ms[camera_index] / static_cast<double>(completed_frames)) << "\n";
            std::cout
                << "  cam" << camera_index
                << "_average_image_process_ms=" << std::fixed << std::setprecision(3)
                << (total_image_process_ms[camera_index] / static_cast<double>(completed_frames)) << "\n";
            std::cout
                << "  cam" << camera_index
                << "_average_worker_ms=" << std::fixed << std::setprecision(3)
                << (total_worker_ms[camera_index] / static_cast<double>(completed_frames)) << "\n";
            std::cout
                << "  cam" << camera_index
                << "_average_frame_bytes=" << (total_frame_bytes[camera_index] / static_cast<std::uint64_t>(completed_frames))
                << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
