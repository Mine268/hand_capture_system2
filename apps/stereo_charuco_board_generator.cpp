#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_board.hpp>

namespace {

struct AppOptions {
    std::string dictionary_name = "DICT_APRILTAG_36h11";
    int squares_x = 7;
    int squares_y = 5;
    float square_length_mm = 40.0f;
    float marker_length_mm = 30.0f;
    int dpi = 300;
    int border_bits = 1;
    std::filesystem::path output_dir = "resources/charuco";
    std::string output_stem;
    float page_width_mm = 210.0f;
    float page_height_mm = 297.0f;
    bool legacy_pattern = false;
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

        if (arg == "--dictionary") options.dictionary_name = require_value(arg);
        else if (arg == "--squares_x") options.squares_x = std::stoi(require_value(arg));
        else if (arg == "--squares_y") options.squares_y = std::stoi(require_value(arg));
        else if (arg == "--square_length_mm") options.square_length_mm = std::stof(require_value(arg));
        else if (arg == "--marker_length_mm") options.marker_length_mm = std::stof(require_value(arg));
        else if (arg == "--dpi") options.dpi = std::stoi(require_value(arg));
        else if (arg == "--border_bits") options.border_bits = std::stoi(require_value(arg));
        else if (arg == "--output_dir") options.output_dir = require_value(arg);
        else if (arg == "--output_stem") options.output_stem = require_value(arg);
        else if (arg == "--page_width_mm") options.page_width_mm = std::stof(require_value(arg));
        else if (arg == "--page_height_mm") options.page_height_mm = std::stof(require_value(arg));
        else if (arg == "--legacy_pattern") options.legacy_pattern = true;
        else if (arg == "--no_legacy_pattern") options.legacy_pattern = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_charuco_board_generator [options]\n"
                << "  --dictionary <name>         default: DICT_APRILTAG_36h11\n"
                << "  --squares_x <int>           default: 7\n"
                << "  --squares_y <int>           default: 5\n"
                << "  --square_length_mm <float>  default: 40\n"
                << "  --marker_length_mm <float>  default: 30\n"
                << "  --dpi <int>                 default: 300\n"
                << "  --border_bits <int>         default: 1\n"
                << "  --output_dir <dir>          default: resources/charuco\n"
                << "  --output_stem <name>        default: auto-generated\n"
                << "  --page_width_mm <float>     default: 210 (A4)\n"
                << "  --page_height_mm <float>    default: 297 (A4)\n"
                << "  --legacy_pattern            default: disabled\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.squares_x < 3 || options.squares_y < 3) {
        throw std::runtime_error("--squares_x and --squares_y must be at least 3");
    }
    if (options.square_length_mm <= 0.0f || options.marker_length_mm <= 0.0f) {
        throw std::runtime_error("--square_length_mm and --marker_length_mm must be positive");
    }
    if (options.marker_length_mm >= options.square_length_mm) {
        throw std::runtime_error("--marker_length_mm must be smaller than --square_length_mm");
    }
    if (options.dpi <= 0) {
        throw std::runtime_error("--dpi must be positive");
    }

    return options;
}

int ParseDictionaryName(const std::string& name) {
    static const std::unordered_map<std::string, int> kDictionaryMap = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
        {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
        {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    const auto it = kDictionaryMap.find(name);
    if (it == kDictionaryMap.end()) {
        throw std::runtime_error("unsupported dictionary: " + name);
    }
    return it->second;
}

std::string DefaultOutputStem(const AppOptions& options) {
    std::ostringstream oss;
    oss
        << "charuco_"
        << options.dictionary_name
        << "_"
        << options.squares_x << "x" << options.squares_y
        << "_sq" << static_cast<int>(std::lround(options.square_length_mm))
        << "mm_mk" << static_cast<int>(std::lround(options.marker_length_mm))
        << "mm";
    return oss.str();
}

void WritePrintableSvg(
    const std::filesystem::path& svg_path,
    const std::string& png_filename,
    const AppOptions& options,
    float board_width_mm,
    float board_height_mm) {
    const float board_x_mm = (options.page_width_mm - board_width_mm) * 0.5f;
    const float board_y_mm = (options.page_height_mm - board_height_mm) * 0.5f;
    if (board_x_mm < 0.0f || board_y_mm < 0.0f) {
        throw std::runtime_error("board size is larger than the target page size");
    }

    const float ruler_x_mm = 20.0f;
    const float ruler_y_mm = options.page_height_mm - 18.0f;
    const float ruler_length_mm = 100.0f;

    std::ofstream out(svg_path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open SVG output: " + svg_path.string());
    }

    out
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << options.page_width_mm << "mm\" "
        << "height=\"" << options.page_height_mm << "mm\" "
        << "viewBox=\"0 0 " << options.page_width_mm << " " << options.page_height_mm << "\">\n"
        << "  <rect x=\"0\" y=\"0\" width=\"" << options.page_width_mm << "\" height=\"" << options.page_height_mm
        << "\" fill=\"white\"/>\n"
        << "  <image href=\"" << png_filename << "\" "
        << "x=\"" << board_x_mm << "\" "
        << "y=\"" << board_y_mm << "\" "
        << "width=\"" << board_width_mm << "\" "
        << "height=\"" << board_height_mm << "\" "
        << "preserveAspectRatio=\"none\"/>\n"
        << "  <line x1=\"" << board_x_mm << "\" y1=\"" << board_y_mm
        << "\" x2=\"" << (board_x_mm + board_width_mm) << "\" y2=\"" << board_y_mm
        << "\" stroke=\"black\" stroke-width=\"0.4\"/>\n"
        << "  <line x1=\"" << board_x_mm << "\" y1=\"" << board_y_mm
        << "\" x2=\"" << board_x_mm << "\" y2=\"" << (board_y_mm + board_height_mm)
        << "\" stroke=\"black\" stroke-width=\"0.4\"/>\n"
        << "  <text x=\"" << (board_x_mm + board_width_mm * 0.5f) << "\" y=\"" << (board_y_mm - 1.5f)
        << "\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\">"
        << board_width_mm << " mm"
        << "</text>\n"
        << "  <text x=\"" << (board_x_mm - 3.0f) << "\" y=\"" << (board_y_mm + board_height_mm * 0.5f)
        << "\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\" transform=\"rotate(-90 "
        << (board_x_mm - 3.0f) << " " << (board_y_mm + board_height_mm * 0.5f) << ")\">"
        << board_height_mm << " mm"
        << "</text>\n"
        << "  <text x=\"" << options.page_width_mm * 0.5f << "\" y=\"" << (board_y_mm - 8.0f)
        << "\" font-size=\"5\" text-anchor=\"middle\" fill=\"black\">"
        << options.dictionary_name
        << " | ChArUco " << options.squares_x << "x" << options.squares_y
        << " | square=" << options.square_length_mm << "mm"
        << " | marker=" << options.marker_length_mm << "mm"
        << "</text>\n"
        << "  <line x1=\"" << ruler_x_mm << "\" y1=\"" << ruler_y_mm
        << "\" x2=\"" << (ruler_x_mm + ruler_length_mm) << "\" y2=\"" << ruler_y_mm
        << "\" stroke=\"black\" stroke-width=\"0.4\"/>\n";

    for (int tick = 0; tick <= static_cast<int>(ruler_length_mm); tick += 10) {
        const float tick_x = ruler_x_mm + static_cast<float>(tick);
        const float tick_height = (tick % 50 == 0) ? 4.5f : 3.0f;
        out
            << "  <line x1=\"" << tick_x << "\" y1=\"" << (ruler_y_mm - tick_height)
            << "\" x2=\"" << tick_x << "\" y2=\"" << (ruler_y_mm + tick_height)
            << "\" stroke=\"black\" stroke-width=\"0.35\"/>\n";
        if (tick < static_cast<int>(ruler_length_mm)) {
            out
                << "  <text x=\"" << tick_x << "\" y=\"" << (ruler_y_mm - 6.0f)
                << "\" font-size=\"3.5\" text-anchor=\"middle\" fill=\"black\">"
                << tick
                << "</text>\n";
        }
    }

    out
        << "  <text x=\"" << (ruler_x_mm + ruler_length_mm * 0.5f) << "\" y=\"" << (ruler_y_mm + 9.0f)
        << "\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\">"
        << "Print check ruler: 100 mm"
        << "</text>\n"
        << "</svg>\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        AppOptions options = ParseArgs(argc, argv);
        if (options.output_stem.empty()) {
            options.output_stem = DefaultOutputStem(options);
        }

        const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(ParseDictionaryName(options.dictionary_name));
        cv::aruco::CharucoBoard board(
            cv::Size(options.squares_x, options.squares_y),
            options.square_length_mm,
            options.marker_length_mm,
            dictionary);
        board.setLegacyPattern(options.legacy_pattern);

        const float board_width_mm = options.squares_x * options.square_length_mm;
        const float board_height_mm = options.squares_y * options.square_length_mm;
        const int out_width_px = static_cast<int>(std::lround(board_width_mm * options.dpi / 25.4f));
        const int out_height_px = static_cast<int>(std::lround(board_height_mm * options.dpi / 25.4f));
        if (out_width_px < 64 || out_height_px < 64) {
            throw std::runtime_error("generated board would be too small in pixels");
        }

        cv::Mat board_image;
        board.generateImage(cv::Size(out_width_px, out_height_px), board_image, 0, options.border_bits);

        std::filesystem::create_directories(options.output_dir);
        const std::filesystem::path png_path = options.output_dir / (options.output_stem + ".png");
        const std::filesystem::path svg_path = options.output_dir / (options.output_stem + "_print.svg");
        if (!cv::imwrite(png_path.string(), board_image)) {
            throw std::runtime_error("failed to write PNG output: " + png_path.string());
        }
        WritePrintableSvg(svg_path, png_path.filename().string(), options, board_width_mm, board_height_mm);

        std::cout << "Generated ChArUco PNG: " << png_path << "\n";
        std::cout << "Generated printable SVG: " << svg_path << "\n";
        std::cout << "Dictionary: " << options.dictionary_name << "\n";
        std::cout << "Board squares: " << options.squares_x << " x " << options.squares_y << "\n";
        std::cout << "Square length: " << options.square_length_mm << " mm\n";
        std::cout << "Marker length: " << options.marker_length_mm << " mm\n";
        std::cout << "Board physical size: " << board_width_mm << " mm x " << board_height_mm << " mm\n";
        std::cout << "Raster size: " << out_width_px << " x " << out_height_px << " px @ " << options.dpi << " DPI\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
