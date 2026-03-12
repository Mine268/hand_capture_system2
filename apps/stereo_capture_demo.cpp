#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/capture/stereo_capture.h"

namespace {

struct DemoOptions {
    std::string output_dir = ".";
    std::string cam0_serial;
    std::string cam1_serial;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 25;
    int frames = -1;
    bool preview = true;
    bool save = false;
};

void PrintEffectiveConfig(
    const DemoOptions& options,
    const newnewhand::StereoCaptureConfig& capture_config) {
    std::cout << "Capture config:\n";
    std::cout << "  output_dir=" << options.output_dir << "\n";
    std::cout << "  cam0_serial=" << (capture_config.serial_numbers[0].empty() ? "<auto>" : capture_config.serial_numbers[0]) << "\n";
    std::cout << "  cam1_serial=" << (capture_config.serial_numbers[1].empty() ? "<auto>" : capture_config.serial_numbers[1]) << "\n";
    std::cout << "  exposure_us=" << capture_config.camera_settings.exposure_us << "\n";
    std::cout << "  gain=" << capture_config.camera_settings.gain << "\n";
    std::cout << "  trigger_timeout_ms=" << capture_config.camera_settings.trigger_timeout_ms << "\n";
    std::cout << "  enable_gamma=" << (capture_config.camera_settings.enable_gamma ? "true" : "false") << "\n";
    std::cout << "  fps=" << options.fps << "\n";
    std::cout << "  frames=" << options.frames << "\n";
    std::cout << "  preview=" << (options.preview ? "true" : "false") << "\n";
    std::cout << "  save=" << (options.save ? "true" : "false") << "\n";
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

        if (arg == "--output_dir") {
            options.output_dir = require_value(arg);
        } else if (arg == "--cam0_serial") {
            options.cam0_serial = require_value(arg);
        } else if (arg == "--cam1_serial") {
            options.cam1_serial = require_value(arg);
        } else if (arg == "--exposure_us") {
            options.exposure_us = std::stof(require_value(arg));
        } else if (arg == "--gain") {
            options.gain = std::stof(require_value(arg));
        } else if (arg == "--fps") {
            options.fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        } else if (arg == "--frames") {
            options.frames = std::stoi(require_value(arg));
        } else if (arg == "--preview") {
            options.preview = true;
        } else if (arg == "--no_preview") {
            options.preview = false;
        } else if (arg == "--save") {
            options.save = true;
        } else if (arg == "--no_save") {
            options.save = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_capture_demo [options]\n"
                << "  --output_dir <dir>        default: .\n"
                << "  --cam0_serial <serial>    default: auto select\n"
                << "  --cam1_serial <serial>    default: auto select\n"
                << "  --exposure_us <float>     default: 10000\n"
                << "  --gain <float>            default: -1 (auto)\n"
                << "  --fps <int>               default: 25\n"
                << "  --frames <int>            default: -1 (run until quit)\n"
                << "  --save | --no_save        default: --no_save\n"
                << "  --preview | --no_preview  default: --preview\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return options;
}

void SaveFrame(
    const std::filesystem::path& root,
    std::size_t camera_index,
    std::uint64_t capture_index,
    const cv::Mat& image) {
    const std::filesystem::path camera_dir = root / ("cam" + std::to_string(camera_index));
    std::filesystem::create_directories(camera_dir);

    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << ".bmp";
    cv::imwrite((camera_dir / name.str()).string(), image);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        DemoOptions options = ParseArgs(argc, argv);

        const auto connected = newnewhand::StereoCapture::EnumerateConnectedCameras();
        std::cout << "Connected cameras: " << connected.size() << "\n";
        for (const auto& descriptor : connected) {
            std::cout
                << "  index=" << descriptor.device_index
                << " serial=" << descriptor.serial_number
                << " model=" << descriptor.model_name
                << "\n";
        }

        newnewhand::StereoCaptureConfig config;
        config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        config.camera_settings.exposure_us = options.exposure_us;
        config.camera_settings.gain = options.gain;
        PrintEffectiveConfig(options, config);

        newnewhand::StereoCapture capture(config);
        capture.Initialize();
        capture.Start();

        const auto active = capture.ActiveCameras();
        std::cout << "Stereo order:\n";
        std::cout << "  cam0 serial=" << active[0].serial_number << " model=" << active[0].model_name << "\n";
        std::cout << "  cam1 serial=" << active[1].serial_number << " model=" << active[1].model_name << "\n";

        if (options.save) {
            std::filesystem::create_directories(options.output_dir);
        }

        const auto frame_interval = options.fps == 0
            ? std::chrono::microseconds(0)
            : std::chrono::microseconds(1000000 / options.fps);

        for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
            const auto loop_start = std::chrono::steady_clock::now();
            newnewhand::StereoFrame stereo_frame = capture.Capture();

            for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
                const auto& frame = stereo_frame.views[camera_index];
                if (!frame.valid) {
                    std::cerr
                        << "capture " << stereo_frame.capture_index
                        << " cam" << camera_index
                        << " failed: " << frame.error_message
                        << "\n";
                    continue;
                }

                if (options.preview) {
                    cv::Mat preview;
                    cv::resize(frame.bgr_image, preview, cv::Size(), 0.5, 0.5);
                    cv::imshow("cam" + std::to_string(camera_index), preview);
                }

                if (options.save) {
                    SaveFrame(options.output_dir, camera_index, stereo_frame.capture_index, frame.bgr_image);
                }
            }

            if (options.preview) {
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    break;
                }
            }

            if (frame_interval.count() > 0) {
                const auto elapsed = std::chrono::steady_clock::now() - loop_start;
                const auto remaining = frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
                if (remaining.count() > 0) {
                    std::this_thread::sleep_for(remaining);
                }
            }
        }

        capture.Stop();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
