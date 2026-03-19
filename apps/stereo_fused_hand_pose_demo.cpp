#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/fusion/stereo_hand_fuser.h"
#include "newnewhand/io/offline_sequence_writer.h"
#include "newnewhand/render/glfw_scene_viewer.h"

namespace {

std::string ProjectRoot() {
#ifdef NEWNEWHAND_PROJECT_ROOT
    return NEWNEWHAND_PROJECT_ROOT;
#else
    return ".";
#endif
}

std::string DefaultDetectorModelPath() { return ProjectRoot() + "/resources/models/detector.onnx"; }
std::string DefaultWilorModelPath() { return ProjectRoot() + "/resources/models/wilor_backbone_opset16.onnx"; }
std::string DefaultManoModelPath() { return ProjectRoot() + "/resources/models/mano_cpu_opset16.onnx"; }
std::string DefaultCalibrationPath() { return ProjectRoot() + "/resources/stereo_calibration.yaml"; }

std::string FormatPoint(float x, float y) {
    std::ostringstream oss;
    oss << "(" << x << ", " << y << ")";
    return oss.str();
}

struct DemoOptions {
    std::string output_dir = "results/stereo_fused_hand_pose";
    std::string offline_dump_dir;
    std::string debug_dir = "debug/wilor_failures";
    std::string ort_profile_prefix;
    std::string calibration_path = DefaultCalibrationPath();
    std::string cam0_serial;
    std::string cam1_serial;
    std::string detector_model_path = DefaultDetectorModelPath();
    std::string wilor_model_path = DefaultWilorModelPath();
    std::string mano_model_path = DefaultManoModelPath();
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 10;
    int frames = -1;
    bool preview = true;
    bool save = true;
    bool use_gpu = true;
    bool verbose = true;
    bool glfw_view = true;
};

struct LatestFrameData {
    newnewhand::StereoSingleViewPoseFrame stereo_frame;
    newnewhand::StereoFusedHandPoseFrame fused_frame;
};

struct SharedWorkerState {
    std::mutex mutex;
    std::shared_ptr<const LatestFrameData> latest_frame;
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

        if (arg == "--output_dir") options.output_dir = require_value(arg);
        else if (arg == "--debug_dir") options.debug_dir = require_value(arg);
        else if (arg == "--offline_dump_dir") options.offline_dump_dir = require_value(arg);
        else if (arg == "--ort_profile") options.ort_profile_prefix = require_value(arg);
        else if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--cam0_serial") options.cam0_serial = require_value(arg);
        else if (arg == "--cam1_serial") options.cam1_serial = require_value(arg);
        else if (arg == "--detector_model") options.detector_model_path = require_value(arg);
        else if (arg == "--wilor_model") options.wilor_model_path = require_value(arg);
        else if (arg == "--mano_model") options.mano_model_path = require_value(arg);
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--fps") options.fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--save") options.save = true;
        else if (arg == "--no_save") options.save = false;
        else if (arg == "--glfw_view") options.glfw_view = true;
        else if (arg == "--no_glfw_view") options.glfw_view = false;
        else if (arg == "--gpu") options.use_gpu = true;
        else if (arg == "--cpu") options.use_gpu = false;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--quiet") options.verbose = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_fused_hand_pose_demo [options]\n"
                << "  --calibration <path>      default: " << DefaultCalibrationPath() << "\n"
                << "  --detector_model <path>   default: " << DefaultDetectorModelPath() << "\n"
                << "  --wilor_model <path>      default: " << DefaultWilorModelPath() << "\n"
                << "  --mano_model <path>       default: " << DefaultManoModelPath() << "\n"
                << "  --output_dir <dir>        default: results/stereo_fused_hand_pose\n"
                << "  --offline_dump_dir <dir>  default: disabled\n"
                << "  --save | --no_save        default: --save\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --glfw_view | --no_glfw_view  default: --glfw_view\n"
                << "  --verbose | --quiet       default: --verbose\n"
                << "  --gpu | --cpu             default: --gpu\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void SaveImage(
    const std::filesystem::path& root,
    const std::string& subdir,
    std::uint64_t capture_index,
    const cv::Mat& image) {
    std::filesystem::create_directories(root / subdir);
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << ".png";
    cv::imwrite((root / subdir / name.str()).string(), image);
}

void SaveFusedYaml(
    const newnewhand::StereoHandFuser& fuser,
    const newnewhand::StereoFusedHandPoseFrame& frame,
    const std::filesystem::path& root) {
    std::filesystem::create_directories(root / "yaml");
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << frame.capture_index << ".yaml";
    fuser.SaveFrame(frame, root / "yaml" / name.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);

        newnewhand::StereoSingleViewHandPosePipelineConfig pipeline_config;
        pipeline_config.capture_config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        pipeline_config.capture_config.camera_settings.exposure_us = options.exposure_us;
        pipeline_config.capture_config.camera_settings.gain = options.gain;
        pipeline_config.pose_config.detector_model_path = options.detector_model_path;
        pipeline_config.pose_config.wilor_model_path = options.wilor_model_path;
        pipeline_config.pose_config.mano_model_path = options.mano_model_path;
        pipeline_config.pose_config.debug_dump_dir = options.debug_dir;
        pipeline_config.pose_config.ort_profile_prefix = options.ort_profile_prefix;
        pipeline_config.pose_config.use_gpu = options.use_gpu;

        const auto calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        newnewhand::StereoHandFuserConfig fuser_config;
        fuser_config.calibration = calibration;
        fuser_config.require_both_views = true;
        fuser_config.verbose_logging = options.verbose;

        newnewhand::GlfwSceneViewerConfig viewer_config;
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
                newnewhand::StereoHandFuser fuser(std::move(fuser_config));
                std::unique_ptr<newnewhand::OfflineSequenceWriter> offline_writer;
                if (!options.offline_dump_dir.empty()) {
                    newnewhand::OfflineSequenceWriterConfig writer_config;
                    writer_config.output_root = options.offline_dump_dir;
                    writer_config.calibration_source_path = options.calibration_path;
                    writer_config.save_raw_images = true;
                    writer_config.save_overlay_images = true;
                    offline_writer = std::make_unique<newnewhand::OfflineSequenceWriter>(writer_config, calibration);
                    offline_writer->Initialize();
                }

                newnewhand::StereoSingleViewHandPosePipeline pipeline(pipeline_config);
                pipeline.Initialize();
                pipeline.Start();

                try {
                    for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
                        if (stop_requested.load()) {
                            break;
                        }

                        const auto loop_start = std::chrono::steady_clock::now();
                        auto stereo_frame = pipeline.CaptureAndEstimate();
                        if (stop_requested.load()) {
                            break;
                        }

                        if (options.verbose) {
                            std::cerr << "[app] capture=" << stereo_frame.capture_index << "\n";
                            for (std::size_t view_index = 0; view_index < stereo_frame.views.size(); ++view_index) {
                                const auto& view = stereo_frame.views[view_index];
                                std::cerr
                                    << "[app] view" << view_index
                                    << " hands=" << view.hand_poses.size()
                                    << " valid=" << view.camera_frame.valid
                                    << " error=" << (view.inference_error.empty() ? "<none>" : view.inference_error)
                                    << "\n";
                                for (std::size_t hand_index = 0; hand_index < view.hand_poses.size(); ++hand_index) {
                                    const auto& hand = view.hand_poses[hand_index];
                                    std::cerr
                                        << "[app] view" << view_index
                                        << " hand" << hand_index
                                        << " type=" << (hand.detection.is_right ? "right" : "left")
                                        << " bbox=(" << hand.detection.bbox[0] << ", " << hand.detection.bbox[1]
                                        << ", " << hand.detection.bbox[2] << ", " << hand.detection.bbox[3] << ")"
                                        << " root2d=" << FormatPoint(hand.keypoints_2d[0][0], hand.keypoints_2d[0][1])
                                        << "\n";
                                }
                            }
                        }

                        auto fused_frame = fuser.Fuse(stereo_frame);
                        std::cout
                            << "capture=" << fused_frame.capture_index
                            << " fused_hands=" << fused_frame.hands.size() << "\n";

                        auto latest_frame = std::make_shared<LatestFrameData>();
                        latest_frame->stereo_frame = stereo_frame;
                        latest_frame->fused_frame = fused_frame;
                        {
                            std::lock_guard<std::mutex> lock(shared_state.mutex);
                            shared_state.latest_frame = std::move(latest_frame);
                        }

                        if (options.save) {
                            for (std::size_t i = 0; i < stereo_frame.views.size(); ++i) {
                                if (!stereo_frame.views[i].overlay_image.empty()) {
                                    SaveImage(
                                        options.output_dir,
                                        "cam" + std::to_string(i),
                                        fused_frame.capture_index,
                                        stereo_frame.views[i].overlay_image);
                                }
                            }
                            SaveFusedYaml(fuser, fused_frame, options.output_dir);
                        }

                        if (offline_writer) {
                            offline_writer->SaveFrame(stereo_frame, fused_frame);
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
                    pipeline.Stop();
                    throw;
                }

                pipeline.Stop();
            } catch (...) {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                shared_state.worker_error = std::current_exception();
            }
            worker_finished.store(true);
        });

        std::shared_ptr<const LatestFrameData> latest_frame;
        std::exception_ptr worker_error;
        const newnewhand::StereoFusedHandPoseFrame empty_fused_frame;

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

            if (options.glfw_view) {
                const auto& fused_frame = latest_frame ? latest_frame->fused_frame : empty_fused_frame;
                if (!viewer.Render(fused_frame)) {
                    std::cerr << "[app] GLFW viewer closed by user\n";
                    stop_requested.store(true);
                    break;
                }
            }

            if (options.preview) {
                if (latest_frame) {
                    for (std::size_t i = 0; i < latest_frame->stereo_frame.views.size(); ++i) {
                        if (!latest_frame->stereo_frame.views[i].overlay_image.empty()) {
                            cv::Mat preview;
                            cv::resize(
                                latest_frame->stereo_frame.views[i].overlay_image,
                                preview,
                                cv::Size(),
                                0.5,
                                0.5);
                            cv::imshow("fused_pose_cam" + std::to_string(i), preview);
                        }
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
