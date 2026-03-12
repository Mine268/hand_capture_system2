#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "newnewhand/calibration/stereo_calibrator.h"

namespace {

struct AppOptions {
    std::filesystem::path left_dir;
    std::filesystem::path right_dir;
    std::filesystem::path output_path = "results/stereo_calibration.yaml";
    std::filesystem::path debug_dir;
    int inner_corners_cols = 0;
    int inner_corners_rows = 0;
    float square_size = 0.0f;
    bool use_find_chessboard_sb = true;
    bool show_detection_preview = false;
    std::size_t max_pairs = 0;
};

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

        if (arg == "--left_dir") {
            options.left_dir = require_value(arg);
        } else if (arg == "--right_dir") {
            options.right_dir = require_value(arg);
        } else if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--debug_dir") {
            options.debug_dir = require_value(arg);
        } else if (arg == "--cols") {
            options.inner_corners_cols = std::stoi(require_value(arg));
        } else if (arg == "--rows") {
            options.inner_corners_rows = std::stoi(require_value(arg));
        } else if (arg == "--square_size") {
            options.square_size = std::stof(require_value(arg));
        } else if (arg == "--max_pairs") {
            options.max_pairs = static_cast<std::size_t>(std::stoul(require_value(arg)));
        } else if (arg == "--use_sb") {
            options.use_find_chessboard_sb = true;
        } else if (arg == "--no_use_sb") {
            options.use_find_chessboard_sb = false;
        } else if (arg == "--preview") {
            options.show_detection_preview = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_calibration_app --left_dir <dir> --right_dir <dir> --cols <int> --rows <int> --square_size <float> [options]\n"
                << "  --output <path>     default: results/stereo_calibration.yaml\n"
                << "  --debug_dir <dir>   save checkerboard detection images\n"
                << "  --max_pairs <int>   default: 0 (use all matched pairs)\n"
                << "  --use_sb            default: enabled, use findChessboardCornersSB\n"
                << "  --no_use_sb         use classic findChessboardCorners + cornerSubPix\n"
                << "  --preview           show checkerboard detection preview windows\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.left_dir.empty() || options.right_dir.empty()) {
        throw std::runtime_error("--left_dir and --right_dir are required");
    }
    if (options.inner_corners_cols <= 0 || options.inner_corners_rows <= 0) {
        throw std::runtime_error("--cols and --rows must be positive");
    }
    if (options.square_size <= 0.0f) {
        throw std::runtime_error("--square_size must be positive");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppOptions options = ParseArgs(argc, argv);

        auto pairs = newnewhand::StereoCalibrator::CollectImagePairs(options.left_dir, options.right_dir);
        if (options.max_pairs > 0 && pairs.size() > options.max_pairs) {
            pairs.resize(options.max_pairs);
        }

        std::cout << "Matched calibration pairs: " << pairs.size() << "\n";
        if (pairs.empty()) {
            throw std::runtime_error("no matched image pairs found");
        }

        newnewhand::StereoCalibrationConfig config;
        config.checkerboard.inner_corners_cols = options.inner_corners_cols;
        config.checkerboard.inner_corners_rows = options.inner_corners_rows;
        config.checkerboard.square_size = options.square_size;
        config.use_find_chessboard_sb = options.use_find_chessboard_sb;
        config.show_detection_preview = options.show_detection_preview;
        config.save_debug_images = !options.debug_dir.empty();
        config.debug_output_dir = options.debug_dir;
        config.max_image_pairs = options.max_pairs;

        newnewhand::StereoCalibrator calibrator(config);
        const auto result = calibrator.Calibrate(pairs);
        calibrator.SaveResult(result, options.output_path);

        std::cout << "Valid calibration pairs: " << result.observations.size() << "\n";
        std::cout << "Image size: " << result.image_size.width << "x" << result.image_size.height << "\n";
        std::cout << "Left RMS: " << result.left_rms << "\n";
        std::cout << "Right RMS: " << result.right_rms << "\n";
        std::cout << "Stereo RMS: " << result.stereo_rms << "\n";
        std::cout << "Saved calibration: " << options.output_path << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
