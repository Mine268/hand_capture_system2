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

#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"

namespace {

constexpr const char* kDefaultDetectorModelPath =
    "/home/renkaiwen/src/wilor_deploy/wilor_deploy/WiLoR-mini/wilor_mini/pretrained_models/detector.onnx";
constexpr const char* kDefaultWilorModelPath =
    "/home/renkaiwen/src/wilor_deploy/wilor_deploy/onnx_model/wilor_safe_rotmat_opset16.onnx";

struct DemoOptions {
    std::string output_dir = "results/stereo_single_view_pose";
    std::string debug_dir = "debug/wilor_failures";
    std::string cam0_serial;
    std::string cam1_serial;
    std::string detector_model_path = kDefaultDetectorModelPath;
    std::string wilor_model_path = kDefaultWilorModelPath;
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 10;
    int frames = -1;
    bool preview = true;
    bool save = false;
    bool use_gpu = true;
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

        if (arg == "--output_dir") {
            options.output_dir = require_value(arg);
        } else if (arg == "--debug_dir") {
            options.debug_dir = require_value(arg);
        } else if (arg == "--cam0_serial") {
            options.cam0_serial = require_value(arg);
        } else if (arg == "--cam1_serial") {
            options.cam1_serial = require_value(arg);
        } else if (arg == "--detector_model") {
            options.detector_model_path = require_value(arg);
        } else if (arg == "--wilor_model") {
            options.wilor_model_path = require_value(arg);
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
        } else if (arg == "--gpu") {
            options.use_gpu = true;
        } else if (arg == "--cpu") {
            options.use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_single_view_hand_pose_demo [options]\n"
                << "  --output_dir <dir>        default: results/stereo_single_view_pose\n"
                << "  --debug_dir <dir>         default: debug/wilor_failures\n"
                << "  --cam0_serial <serial>    default: auto select\n"
                << "  --cam1_serial <serial>    default: auto select\n"
                << "  --detector_model <path>   default: " << kDefaultDetectorModelPath << "\n"
                << "  --wilor_model <path>      default: " << kDefaultWilorModelPath << "\n"
                << "  --exposure_us <float>     default: 10000\n"
                << "  --gain <float>            default: -1 (auto)\n"
                << "  --fps <int>               default: 10\n"
                << "  --frames <int>            default: -1 (run until quit)\n"
                << "  --save | --no_save        default: --no_save\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --gpu | --cpu            default: --gpu\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return options;
}

void PrintEffectiveConfig(const DemoOptions& options) {
    std::cout << "Stereo single-view pose config:\n";
    std::cout << "  output_dir=" << options.output_dir << "\n";
    std::cout << "  debug_dir=" << options.debug_dir << "\n";
    std::cout << "  cam0_serial=" << (options.cam0_serial.empty() ? "<auto>" : options.cam0_serial) << "\n";
    std::cout << "  cam1_serial=" << (options.cam1_serial.empty() ? "<auto>" : options.cam1_serial) << "\n";
    std::cout << "  detector_model=" << options.detector_model_path << "\n";
    std::cout << "  wilor_model=" << options.wilor_model_path << "\n";
    std::cout << "  exposure_us=" << options.exposure_us << "\n";
    std::cout << "  gain=" << options.gain << "\n";
    std::cout << "  fps=" << options.fps << "\n";
    std::cout << "  frames=" << options.frames << "\n";
    std::cout << "  preview=" << (options.preview ? "true" : "false") << "\n";
    std::cout << "  save=" << (options.save ? "true" : "false") << "\n";
    std::cout << "  use_gpu=" << (options.use_gpu ? "true" : "false") << "\n";
}

void SaveOverlayFrame(
    const std::filesystem::path& root,
    std::size_t camera_index,
    std::uint64_t capture_index,
    const cv::Mat& image) {
    const std::filesystem::path camera_dir = root / ("cam" + std::to_string(camera_index));
    std::filesystem::create_directories(camera_dir);

    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << ".png";
    cv::imwrite((camera_dir / name.str()).string(), image);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);
        PrintEffectiveConfig(options);

        const auto connected_cameras = newnewhand::StereoCapture::EnumerateConnectedCameras();
        std::cout << "Connected cameras: " << connected_cameras.size() << "\n";
        for (const auto& descriptor : connected_cameras) {
            std::cout
                << "  index=" << descriptor.device_index
                << " serial=" << descriptor.serial_number
                << " model=" << descriptor.model_name
                << "\n";
        }

        newnewhand::StereoSingleViewHandPosePipelineConfig config;
        config.capture_config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        config.capture_config.camera_settings.exposure_us = options.exposure_us;
        config.capture_config.camera_settings.gain = options.gain;
        config.pose_config.detector_model_path = options.detector_model_path;
        config.pose_config.wilor_model_path = options.wilor_model_path;
        config.pose_config.debug_dump_dir = options.debug_dir;
        config.pose_config.use_gpu = options.use_gpu;

        newnewhand::StereoSingleViewHandPosePipeline pipeline(config);
        pipeline.Initialize();
        pipeline.Start();

        const auto active = pipeline.ActiveCameras();
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
            const auto pose_frame = pipeline.CaptureAndEstimate();

            std::cout << "capture=" << pose_frame.capture_index;
            for (std::size_t camera_index = 0; camera_index < pose_frame.views.size(); ++camera_index) {
                const auto& view = pose_frame.views[camera_index];
                std::cout << " cam" << camera_index << "_hands=" << view.hand_poses.size();
                if (view.used_cpu_fallback) {
                    std::cout << "(cpu_fallback)";
                }
                if (!view.inference_error.empty()) {
                    std::cout << "(error)";
                }
            }
            std::cout << "\n";

            for (std::size_t camera_index = 0; camera_index < pose_frame.views.size(); ++camera_index) {
                const auto& view = pose_frame.views[camera_index];
                if (!view.inference_error.empty()) {
                    std::cerr
                        << "cam" << camera_index
                        << " inference error: " << view.inference_error
                        << "\n";
                }
            }

            for (std::size_t camera_index = 0; camera_index < pose_frame.views.size(); ++camera_index) {
                const auto& view = pose_frame.views[camera_index];
                if (!view.camera_frame.valid || view.overlay_image.empty()) {
                    continue;
                }

                if (options.preview) {
                    cv::Mat preview;
                    cv::resize(view.overlay_image, preview, cv::Size(), 0.5, 0.5);
                    cv::imshow("pose_cam" + std::to_string(camera_index), preview);
                }

                if (options.save) {
                    SaveOverlayFrame(
                        options.output_dir,
                        camera_index,
                        pose_frame.capture_index,
                        view.overlay_image);
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
                const auto remaining =
                    frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
                if (remaining.count() > 0) {
                    std::this_thread::sleep_for(remaining);
                }
            }
        }

        pipeline.Stop();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
