#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "newnewhand/perception/hand_pose_estimator.h"
#include "newnewhand/visualization/hand_pose_overlay.h"

namespace {

std::string ProjectRoot() {
#ifdef NEWNEWHAND_PROJECT_ROOT
    return NEWNEWHAND_PROJECT_ROOT;
#else
    return ".";
#endif
}

std::string DefaultDetectorModelPath() {
    return ProjectRoot() + "/resources/models/detector.onnx";
}

std::string DefaultWilorModelPath() {
    return ProjectRoot() + "/resources/models/wilor_backbone_opset16.onnx";
}

std::string DefaultManoModelPath() {
    return ProjectRoot() + "/resources/models/mano_cpu_opset16.onnx";
}

struct DemoOptions {
    std::string image_path;
    std::string detector_model_path = DefaultDetectorModelPath();
    std::string wilor_model_path = DefaultWilorModelPath();
    std::string mano_model_path = DefaultManoModelPath();
    std::string output_path = "output.png";
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

        if (arg == "--image") {
            options.image_path = require_value(arg);
        } else if (arg == "--detector_model") {
            options.detector_model_path = require_value(arg);
        } else if (arg == "--wilor_model") {
            options.wilor_model_path = require_value(arg);
        } else if (arg == "--mano_model") {
            options.mano_model_path = require_value(arg);
        } else if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--gpu") {
            options.use_gpu = true;
        } else if (arg == "--cpu") {
            options.use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: single_view_hand_pose_demo --image <path> [options]\n"
                << "  --detector_model <path> default: " << DefaultDetectorModelPath() << "\n"
                << "  --wilor_model <path>    default: " << DefaultWilorModelPath() << "\n"
                << "  --mano_model <path>     default: " << DefaultManoModelPath() << "\n"
                << "  --output <path>         default: output.png\n"
                << "  --gpu                   default: enabled\n"
                << "  --cpu                   force CPU for detector/backbone\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.image_path.empty()) {
        throw std::runtime_error("--image is required");
    }

    return options;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);

        newnewhand::HandPoseEstimatorConfig config;
        config.detector_model_path = options.detector_model_path;
        config.wilor_model_path = options.wilor_model_path;
        config.mano_model_path = options.mano_model_path;
        config.use_gpu = options.use_gpu;

        const auto load_start = std::chrono::high_resolution_clock::now();
        newnewhand::HandPoseEstimator estimator(config);
        const auto load_end = std::chrono::high_resolution_clock::now();

        const cv::Mat image = cv::imread(options.image_path);
        if (image.empty()) {
            throw std::runtime_error("failed to read image: " + options.image_path);
        }

        const auto infer_start = std::chrono::high_resolution_clock::now();
        const auto results = estimator.Predict(image);
        const auto infer_end = std::chrono::high_resolution_clock::now();

        std::cout << "Image: " << options.image_path << " (" << image.cols << "x" << image.rows << ")\n";
        std::cout << "Model load: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count()
                  << " ms\n";
        std::cout << "Inference: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count()
                  << " ms\n";
        std::cout << "Detected hands: " << results.size() << "\n";

        cv::Mat visualization = newnewhand::RenderHandPoseOverlay(image, results);
        cv::imwrite(options.output_path, visualization);
        std::cout << "Saved visualization: " << options.output_path << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
