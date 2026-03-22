#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "newnewhand/capture/stereo_capture.h"

namespace {

struct DemoOptions {
    std::string cam0_serial;
    std::string cam1_serial;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    int frames = 300;
    bool verbose = false;
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

        if (arg == "--cam0_serial") options.cam0_serial = require_value(arg);
        else if (arg == "--cam1_serial") options.cam1_serial = require_value(arg);
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
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
                << "  --verbose | --quiet       default: --quiet\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.frames <= 0) {
        throw std::runtime_error("--frames must be positive");
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

        newnewhand::StereoCapture capture(config);
        capture.Initialize();
        capture.Start();
        const auto active = capture.ActiveCameras();

        std::cout << "Stereo benchmark order:\n";
        std::cout << "  cam0 serial=" << active[0].serial_number << " model=" << active[0].model_name << "\n";
        std::cout << "  cam1 serial=" << active[1].serial_number << " model=" << active[1].model_name << "\n";

        const auto benchmark_start = std::chrono::steady_clock::now();
        auto previous_trigger_time = std::chrono::steady_clock::time_point{};
        double ema_fps = 0.0;
        double total_capture_ms = 0.0;
        int valid_stereo_frames = 0;

        for (int frame_index = 0; frame_index < options.frames; ++frame_index) {
            const auto capture_start = std::chrono::steady_clock::now();
            const auto stereo_frame = capture.Capture();
            const auto capture_end = std::chrono::steady_clock::now();

            const double capture_ms =
                std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
            total_capture_ms += capture_ms;

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

            const bool valid_stereo = stereo_frame.is_complete();
            if (valid_stereo) {
                valid_stereo_frames += 1;
            }

            if (options.verbose) {
                std::cout
                    << "capture=" << stereo_frame.capture_index
                    << " valid=" << (valid_stereo ? 1 : 0)
                    << " capture_ms=" << std::fixed << std::setprecision(3) << capture_ms
                    << " fps=" << std::fixed << std::setprecision(2) << ema_fps
                    << "\n";
            }
        }

        const auto benchmark_end = std::chrono::steady_clock::now();
        const double total_seconds =
            std::chrono::duration<double>(benchmark_end - benchmark_start).count();
        const double average_fps = options.frames / std::max(1e-6, total_seconds);
        const double average_capture_ms = total_capture_ms / static_cast<double>(options.frames);

        capture.Stop();

        std::cout << "Stereo capture benchmark summary:\n";
        std::cout << "  requested_frames=" << options.frames << "\n";
        std::cout << "  valid_stereo_frames=" << valid_stereo_frames << "\n";
        std::cout << "  total_seconds=" << std::fixed << std::setprecision(3) << total_seconds << "\n";
        std::cout << "  average_fps=" << std::fixed << std::setprecision(2) << average_fps << "\n";
        std::cout << "  ema_fps_last=" << std::fixed << std::setprecision(2) << ema_fps << "\n";
        std::cout << "  average_capture_ms=" << std::fixed << std::setprecision(3) << average_capture_ms << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
