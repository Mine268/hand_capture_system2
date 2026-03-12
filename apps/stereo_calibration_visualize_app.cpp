#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "newnewhand/calibration/stereo_calibration_visualizer.h"

namespace {

struct AppOptions {
    std::filesystem::path calibration_path;
    std::filesystem::path output_image = "results/stereo_calibration_scene.png";
    std::size_t max_pairs = 0;
    bool use_find_chessboard_sb = true;
    bool preview = true;
    bool save = true;
    bool draw_board_ids = false;
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

        if (arg == "--calibration") {
            options.calibration_path = require_value(arg);
        } else if (arg == "--output") {
            options.output_image = require_value(arg);
        } else if (arg == "--max_pairs") {
            options.max_pairs = static_cast<std::size_t>(std::stoul(require_value(arg)));
        } else if (arg == "--use_sb") {
            options.use_find_chessboard_sb = true;
        } else if (arg == "--no_use_sb") {
            options.use_find_chessboard_sb = false;
        } else if (arg == "--preview") {
            options.preview = true;
        } else if (arg == "--no_preview") {
            options.preview = false;
        } else if (arg == "--save") {
            options.save = true;
        } else if (arg == "--no_save") {
            options.save = false;
        } else if (arg == "--draw_ids") {
            options.draw_board_ids = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_calibration_visualize_app --calibration <yaml> [options]\n"
                << "  --output <path>     default: results/stereo_calibration_scene.png\n"
                << "  --max_pairs <int>   default: 0 (use all valid pairs)\n"
                << "  --use_sb            default: enabled\n"
                << "  --no_use_sb         use classic chessboard detector for pose reconstruction\n"
                << "  --draw_ids          draw board indices in the scene\n"
                << "  --preview | --no_preview  default: --preview\n"
                << "  --save | --no_save        default: --save\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.calibration_path.empty()) {
        throw std::runtime_error("--calibration is required");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppOptions options = ParseArgs(argc, argv);
        const auto calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        const auto board_poses = newnewhand::StereoCalibrationVisualizer::EstimateBoardPoses(
            calibration,
            options.use_find_chessboard_sb,
            options.max_pairs);

        newnewhand::StereoCalibrationViewStyle style;
        style.draw_board_ids = options.draw_board_ids;
        const cv::Mat scene = newnewhand::StereoCalibrationVisualizer::RenderScene(calibration, board_poses, style);

        std::cout << "Loaded calibration: " << options.calibration_path << "\n";
        std::cout << "Boards reconstructed: " << board_poses.size() << "\n";

        if (options.save) {
            if (!options.output_image.parent_path().empty()) {
                std::filesystem::create_directories(options.output_image.parent_path());
            }
            cv::imwrite(options.output_image.string(), scene);
            std::cout << "Saved scene image: " << options.output_image << "\n";
        }

        if (options.preview) {
            cv::imshow("stereo_calibration_scene", scene);
            while (true) {
                const int key = cv::waitKey(30);
                if (key == 'q' || key == 27) {
                    break;
                }
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
