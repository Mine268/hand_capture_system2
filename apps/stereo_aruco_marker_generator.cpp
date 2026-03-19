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

namespace {

struct AppOptions {
    std::string dictionary_name = "DICT_APRILTAG_36h11";
    int marker_id = 0;
    float size_mm = 150.0f;
    int dpi = 300;
    int border_bits = 1;
    std::filesystem::path output_dir = "resources/markers";
    std::string output_stem;
    float page_width_mm = 210.0f;
    float page_height_mm = 297.0f;
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

        if (arg == "--dictionary") {
            options.dictionary_name = require_value(arg);
        } else if (arg == "--marker_id") {
            options.marker_id = std::stoi(require_value(arg));
        } else if (arg == "--size_mm") {
            options.size_mm = std::stof(require_value(arg));
        } else if (arg == "--dpi") {
            options.dpi = std::stoi(require_value(arg));
        } else if (arg == "--border_bits") {
            options.border_bits = std::stoi(require_value(arg));
        } else if (arg == "--output_dir") {
            options.output_dir = require_value(arg);
        } else if (arg == "--output_stem") {
            options.output_stem = require_value(arg);
        } else if (arg == "--page_width_mm") {
            options.page_width_mm = std::stof(require_value(arg));
        } else if (arg == "--page_height_mm") {
            options.page_height_mm = std::stof(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_aruco_marker_generator [options]\n"
                << "  --dictionary <name>       default: DICT_APRILTAG_36h11\n"
                << "  --marker_id <int>         default: 0\n"
                << "  --size_mm <float>         default: 150\n"
                << "  --dpi <int>               default: 300\n"
                << "  --border_bits <int>       default: 1\n"
                << "  --output_dir <dir>        default: resources/markers\n"
                << "  --output_stem <name>      default: auto-generated\n"
                << "  --page_width_mm <float>   default: 210 (A4)\n"
                << "  --page_height_mm <float>  default: 297 (A4)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.marker_id < 0) {
        throw std::runtime_error("--marker_id must be non-negative");
    }
    if (options.size_mm <= 0.0f) {
        throw std::runtime_error("--size_mm must be positive");
    }
    if (options.dpi <= 0) {
        throw std::runtime_error("--dpi must be positive");
    }
    if (options.border_bits <= 0) {
        throw std::runtime_error("--border_bits must be positive");
    }
    if (options.page_width_mm <= 0.0f || options.page_height_mm <= 0.0f) {
        throw std::runtime_error("--page_width_mm and --page_height_mm must be positive");
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
        << options.dictionary_name
        << "_id" << options.marker_id
        << "_"
        << static_cast<int>(std::lround(options.size_mm))
        << "mm";
    return oss.str();
}

void WritePrintableSvg(
    const std::filesystem::path& svg_path,
    const std::string& png_filename,
    const AppOptions& options) {
    const float marker_x_mm = (options.page_width_mm - options.size_mm) * 0.5f;
    const float marker_y_mm = (options.page_height_mm - options.size_mm) * 0.5f;
    if (marker_x_mm < 0.0f || marker_y_mm < 0.0f) {
        throw std::runtime_error("marker size is larger than the target page size");
    }

    std::ofstream out(svg_path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open SVG output: " + svg_path.string());
    }

    const float ruler_x_mm = 20.0f;
    const float ruler_y_mm = options.page_height_mm - 18.0f;
    const float ruler_length_mm = 100.0f;

    out
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << options.page_width_mm << "mm\" "
        << "height=\"" << options.page_height_mm << "mm\" "
        << "viewBox=\"0 0 " << options.page_width_mm << " " << options.page_height_mm << "\">\n"
        << "  <rect x=\"0\" y=\"0\" width=\"" << options.page_width_mm << "\" height=\"" << options.page_height_mm
        << "\" fill=\"white\"/>\n"
        << "  <image href=\"" << png_filename << "\" "
        << "x=\"" << marker_x_mm << "\" "
        << "y=\"" << marker_y_mm << "\" "
        << "width=\"" << options.size_mm << "\" "
        << "height=\"" << options.size_mm << "\" "
        << "preserveAspectRatio=\"none\"/>\n"
        << "  <line x1=\"" << marker_x_mm << "\" y1=\"" << marker_y_mm
        << "\" x2=\"" << (marker_x_mm + options.size_mm) << "\" y2=\"" << marker_y_mm
        << "\" stroke=\"black\" stroke-width=\"0.4\"/>\n"
        << "  <line x1=\"" << marker_x_mm << "\" y1=\"" << marker_y_mm
        << "\" x2=\"" << marker_x_mm << "\" y2=\"" << (marker_y_mm + options.size_mm)
        << "\" stroke=\"black\" stroke-width=\"0.4\"/>\n"
        << "  <text x=\"" << options.page_width_mm * 0.5f << "\" y=\"" << (marker_y_mm - 6.0f)
        << "\" font-size=\"5\" text-anchor=\"middle\" fill=\"black\">"
        << options.dictionary_name << " id=" << options.marker_id
        << " size=" << options.size_mm << "mm"
        << "</text>\n"
        << "  <text x=\"" << (marker_x_mm + options.size_mm * 0.5f) << "\" y=\"" << (marker_y_mm - 1.5f)
        << "\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\">"
        << options.size_mm << " mm"
        << "</text>\n"
        << "  <text x=\"" << (marker_x_mm - 3.0f) << "\" y=\"" << (marker_y_mm + options.size_mm * 0.5f)
        << "\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\" transform=\"rotate(-90 "
        << (marker_x_mm - 3.0f) << " " << (marker_y_mm + options.size_mm * 0.5f) << ")\">"
        << options.size_mm << " mm"
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

        const int side_pixels = static_cast<int>(std::lround(options.size_mm * options.dpi / 25.4f));
        if (side_pixels < 64) {
            throw std::runtime_error("generated marker would be too small in pixels");
        }

        const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(ParseDictionaryName(options.dictionary_name));
        cv::Mat marker;
        dictionary.generateImageMarker(options.marker_id, side_pixels, marker, options.border_bits);

        std::filesystem::create_directories(options.output_dir);
        const std::filesystem::path png_path = options.output_dir / (options.output_stem + ".png");
        const std::filesystem::path svg_path = options.output_dir / (options.output_stem + "_print.svg");

        if (!cv::imwrite(png_path.string(), marker)) {
            throw std::runtime_error("failed to write PNG output: " + png_path.string());
        }
        WritePrintableSvg(svg_path, png_path.filename().string(), options);

        std::cout << "Generated marker PNG: " << png_path << "\n";
        std::cout << "Generated printable SVG: " << svg_path << "\n";
        std::cout << "Dictionary: " << options.dictionary_name << "\n";
        std::cout << "Marker ID: " << options.marker_id << "\n";
        std::cout << "Physical size: " << options.size_mm << " mm\n";
        std::cout << "Raster size: " << side_pixels << " x " << side_pixels << " px @ " << options.dpi << " DPI\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
