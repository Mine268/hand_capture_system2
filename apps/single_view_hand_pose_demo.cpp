#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "newnewhand/perception/hand_pose_estimator.h"

namespace {

constexpr int kHandConnections[20][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    {0, 17}, {17, 18}, {18, 19}, {19, 20},
};

const cv::Scalar kFingerColors[5] = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
};

struct DemoOptions {
    std::string image_path;
    std::string detector_model_path;
    std::string wilor_model_path;
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
        } else if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--gpu") {
            options.use_gpu = true;
        } else if (arg == "--cpu") {
            options.use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: single_view_hand_pose_demo --image <path> --detector_model <path> --wilor_model <path> [options]\n"
                << "  --output <path>   default: output.png\n"
                << "  --gpu             default: enabled\n"
                << "  --cpu             force CPU for WiLoR\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.image_path.empty()) {
        throw std::runtime_error("--image is required");
    }
    if (options.detector_model_path.empty()) {
        throw std::runtime_error("--detector_model is required");
    }
    if (options.wilor_model_path.empty()) {
        throw std::runtime_error("--wilor_model is required");
    }

    return options;
}

void DrawResults(cv::Mat& image, const std::vector<newnewhand::HandPoseResult>& results) {
    for (const auto& result : results) {
        cv::rectangle(
            image,
            cv::Point(static_cast<int>(result.detection.bbox[0]), static_cast<int>(result.detection.bbox[1])),
            cv::Point(static_cast<int>(result.detection.bbox[2]), static_cast<int>(result.detection.bbox[3])),
            cv::Scalar(0, 255, 0),
            2);
        const std::string label = result.detection.is_right ? "R" : "L";
        cv::putText(
            image,
            label,
            cv::Point(static_cast<int>(result.detection.bbox[0]), static_cast<int>(result.detection.bbox[1]) - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2);

        for (int connection_index = 0; connection_index < 20; ++connection_index) {
            const int start_joint = kHandConnections[connection_index][0];
            const int end_joint = kHandConnections[connection_index][1];
            cv::line(
                image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[start_joint][0]),
                    static_cast<int>(result.keypoints_2d[start_joint][1])),
                cv::Point(
                    static_cast<int>(result.keypoints_2d[end_joint][0]),
                    static_cast<int>(result.keypoints_2d[end_joint][1])),
                kFingerColors[connection_index / 4],
                2);
        }

        for (int joint_index = 0; joint_index < 21; ++joint_index) {
            cv::circle(
                image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[joint_index][0]),
                    static_cast<int>(result.keypoints_2d[joint_index][1])),
                3,
                cv::Scalar(0, 0, 255),
                -1);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);

        newnewhand::HandPoseEstimatorConfig config;
        config.detector_model_path = options.detector_model_path;
        config.wilor_model_path = options.wilor_model_path;
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

        cv::Mat visualization = image.clone();
        DrawResults(visualization, results);
        cv::imwrite(options.output_path, visualization);
        std::cout << "Saved visualization: " << options.output_path << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
