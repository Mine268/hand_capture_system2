#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/capture/stereo_capture.h"

namespace {

struct DemoOptions {
    std::string mode = "capture";
    std::filesystem::path input_dir;
    std::filesystem::path output_dir = "offline_capture/session_001";
    std::string calibration_path;
    std::string cam0_serial;
    std::string cam1_serial;
    std::string image_format = "png";
    std::string video_codec = "MJPG";
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    unsigned int fps = 30;
    int frames = -1;
    bool preview = true;
    std::size_t writer_queue_size = 256;
};

struct SavedFrameRecord {
    std::uint64_t capture_index = 0;
    std::int64_t trigger_time_us = 0;
    std::array<std::uint64_t, 2> frame_indices = {0, 0};
    std::array<std::uint64_t, 2> device_timestamps = {0, 0};
    std::array<std::int64_t, 2> sdk_host_timestamps = {0, 0};
    std::array<std::int64_t, 2> host_times_us = {0, 0};
    std::array<bool, 2> valid = {false, false};
    std::array<std::int64_t, 2> video_frame_indices = {-1, -1};
};

struct StereoVideoWriteJob {
    std::array<cv::Mat, 2> images;
    std::array<bool, 2> valid = {false, false};
};

struct PreviewFrame {
    std::uint64_t capture_index = 0;
    std::array<cv::Mat, 2> images;
    std::array<bool, 2> valid = {false, false};
    double capture_fps = 0.0;
};

struct SharedPreviewState {
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<const PreviewFrame> latest_frame;
    bool stop_requested = false;
    bool quit_requested = false;
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

        if (arg == "--mode") options.mode = require_value(arg);
        else if (arg == "--input_dir") options.input_dir = require_value(arg);
        else if (arg == "--output_dir") options.output_dir = require_value(arg);
        else if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--cam0_serial") options.cam0_serial = require_value(arg);
        else if (arg == "--cam1_serial") options.cam1_serial = require_value(arg);
        else if (arg == "--image_format") options.image_format = require_value(arg);
        else if (arg == "--video_codec") options.video_codec = require_value(arg);
        else if (arg == "--exposure_us") options.exposure_us = std::stof(require_value(arg));
        else if (arg == "--gain") options.gain = std::stof(require_value(arg));
        else if (arg == "--fps") options.fps = static_cast<unsigned int>(std::stoul(require_value(arg)));
        else if (arg == "--frames") options.frames = std::stoi(require_value(arg));
        else if (arg == "--writer_queue") options.writer_queue_size = static_cast<std::size_t>(std::stoull(require_value(arg)));
        else if (arg == "--preview") options.preview = true;
        else if (arg == "--no_preview") options.preview = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stereo_offline_capture_app [options]\n"
                << "  --mode <capture|decode>   default: capture\n"
                << "  --input_dir <dir>         required in decode mode\n"
                << "  --output_dir <dir>        default: offline_capture/session_001\n"
                << "  --calibration <path>      default: disabled\n"
                << "  --cam0_serial <serial>    default: auto select\n"
                << "  --cam1_serial <serial>    default: auto select\n"
                << "  --image_format <fmt>      default: png (png|bmp|jpg)\n"
                << "  --video_codec <fourcc>    default: MJPG\n"
                << "  --exposure_us <float>     default: 10000\n"
                << "  --gain <float>            default: -1 (auto)\n"
                << "  --fps <int>               default: 30\n"
                << "  --frames <int>            default: -1 (run until quit / use all frames)\n"
                << "  --writer_queue <int>      default: 256\n"
                << "  --preview | --no_preview  default: --preview\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.mode != "capture" && options.mode != "decode") {
        throw std::runtime_error("--mode must be capture or decode");
    }
    if (options.fps == 0) {
        throw std::runtime_error("--fps must be positive");
    }
    if (options.writer_queue_size == 0) {
        throw std::runtime_error("--writer_queue must be positive");
    }
    if (options.image_format != "png" && options.image_format != "bmp" && options.image_format != "jpg") {
        throw std::runtime_error("--image_format must be one of: png, bmp, jpg");
    }
    if (options.video_codec.size() != 4) {
        throw std::runtime_error("--video_codec must be a four-character code such as MJPG");
    }
    if (options.mode == "decode" && options.input_dir.empty()) {
        throw std::runtime_error("--input_dir is required in decode mode");
    }
    return options;
}

void ApplyCalibrationSerials(
    const DemoOptions& options,
    const newnewhand::StereoCalibrationResult& calibration,
    newnewhand::StereoCaptureConfig& capture_config) {
    const bool has_saved_serials =
        !calibration.left_camera_serial_number.empty() && !calibration.right_camera_serial_number.empty();
    if (!has_saved_serials) {
        capture_config.serial_numbers = {options.cam0_serial, options.cam1_serial};
        return;
    }

    if (!options.cam0_serial.empty() && options.cam0_serial != calibration.left_camera_serial_number) {
        throw std::runtime_error(
            "requested --cam0_serial does not match calibration left_camera_serial_number");
    }
    if (!options.cam1_serial.empty() && options.cam1_serial != calibration.right_camera_serial_number) {
        throw std::runtime_error(
            "requested --cam1_serial does not match calibration right_camera_serial_number");
    }

    capture_config.serial_numbers = {
        calibration.left_camera_serial_number,
        calibration.right_camera_serial_number,
    };
}

std::filesystem::path BuildVideoPath(
    const std::filesystem::path& output_root,
    std::size_t camera_index) {
    return output_root / "videos" / ("cam" + std::to_string(camera_index) + ".avi");
}

std::filesystem::path BuildFrameImagePath(
    const std::filesystem::path& output_root,
    std::size_t camera_index,
    std::uint64_t capture_index,
    const std::string& image_format) {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << capture_index << "." << image_format;
    return output_root / "images" / ("cam" + std::to_string(camera_index)) / name.str();
}

void UpdatePreviewWindowTitle(
    std::size_t camera_index,
    const newnewhand::CameraDescriptor& descriptor,
    double capture_fps) {
    std::ostringstream title;
    title
        << "offline_capture_cam" << camera_index
        << " | serial " << descriptor.serial_number
        << " | capture "
        << std::fixed << std::setprecision(1) << capture_fps
        << " FPS";
    cv::setWindowTitle("offline_capture_cam" + std::to_string(camera_index), title.str());
}

void PreviewWorker(
    const std::array<newnewhand::CameraDescriptor, 2>& active_cameras,
    SharedPreviewState& preview_state) {
    while (true) {
        std::shared_ptr<const PreviewFrame> frame;
        {
            std::unique_lock<std::mutex> lock(preview_state.mutex);
            preview_state.cv.wait(lock, [&]() {
                return preview_state.stop_requested || preview_state.latest_frame != nullptr;
            });
            if (preview_state.stop_requested && preview_state.latest_frame == nullptr) {
                break;
            }
            frame = preview_state.latest_frame;
            preview_state.latest_frame.reset();
        }

        if (!frame) {
            continue;
        }

        for (std::size_t camera_index = 0; camera_index < frame->images.size(); ++camera_index) {
            if (!frame->valid[camera_index] || frame->images[camera_index].empty()) {
                continue;
            }
            cv::Mat preview;
            cv::resize(frame->images[camera_index], preview, cv::Size(), 0.5, 0.5);
            cv::imshow("offline_capture_cam" + std::to_string(camera_index), preview);
            UpdatePreviewWindowTitle(camera_index, active_cameras[camera_index], frame->capture_fps);
        }
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            std::lock_guard<std::mutex> lock(preview_state.mutex);
            preview_state.quit_requested = true;
            preview_state.stop_requested = true;
            preview_state.latest_frame.reset();
            preview_state.cv.notify_all();
            break;
        }
    }
}

class AsyncStereoVideoWriter {
public:
    AsyncStereoVideoWriter(
        const std::filesystem::path& output_root,
        int fourcc,
        double fps,
        std::size_t max_queue_size)
        : output_root_(output_root),
          fourcc_(fourcc),
          fps_(fps),
          max_queue_size_(max_queue_size),
          worker_([this]() { WorkerLoop(); }) {}

    ~AsyncStereoVideoWriter() {
        try {
            Close();
        } catch (...) {
        }
    }

    void Enqueue(StereoVideoWriteJob job) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&]() {
            return closed_ || queue_.size() < max_queue_size_ || worker_error_;
        });
        if (worker_error_) {
            std::rethrow_exception(worker_error_);
        }
        if (closed_) {
            throw std::runtime_error("async video writer is already closed");
        }
        queue_.push_back(std::move(job));
        cv_not_empty_.notify_one();
    }

    void Close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                if (worker_error_) {
                    std::rethrow_exception(worker_error_);
                }
                return;
            }
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (worker_error_) {
            std::rethrow_exception(worker_error_);
        }
    }

private:
    void EnsureWritersOpened(const StereoVideoWriteJob& job) {
        if (writers_opened_) {
            return;
        }

        cv::Size frame_size;
        for (std::size_t camera_index = 0; camera_index < job.images.size(); ++camera_index) {
            if (job.valid[camera_index] && !job.images[camera_index].empty()) {
                frame_size = job.images[camera_index].size();
                break;
            }
        }
        if (frame_size.width <= 0 || frame_size.height <= 0) {
            throw std::runtime_error("cannot initialize video writer from empty stereo frame");
        }

        std::filesystem::create_directories(output_root_ / "videos");
        for (std::size_t camera_index = 0; camera_index < 2; ++camera_index) {
            const std::filesystem::path output_path = BuildVideoPath(output_root_, camera_index);
            if (!writers_[camera_index].open(output_path.string(), fourcc_, fps_, frame_size, true)) {
                throw std::runtime_error("failed to open video writer: " + output_path.string());
            }
            blank_frames_[camera_index] = cv::Mat::zeros(frame_size, CV_8UC3);
        }
        writers_opened_ = true;
    }

    void WorkerLoop() {
        try {
            while (true) {
                StereoVideoWriteJob job;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_not_empty_.wait(lock, [&]() {
                        return closed_ || !queue_.empty();
                    });
                    if (queue_.empty()) {
                        return;
                    }
                    job = std::move(queue_.front());
                    queue_.pop_front();
                    cv_not_full_.notify_one();
                }

                EnsureWritersOpened(job);
                for (std::size_t camera_index = 0; camera_index < 2; ++camera_index) {
                    const cv::Mat& frame =
                        job.valid[camera_index] && !job.images[camera_index].empty()
                        ? job.images[camera_index]
                        : blank_frames_[camera_index];
                    writers_[camera_index].write(frame);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!worker_error_) {
                worker_error_ = std::current_exception();
            }
            closed_ = true;
            cv_not_empty_.notify_all();
            cv_not_full_.notify_all();
        }
    }

    std::filesystem::path output_root;
    std::filesystem::path output_root_;
    int fourcc_ = 0;
    double fps_ = 0.0;
    std::size_t max_queue_size_ = 0;
    bool closed_ = false;
    bool writers_opened_ = false;
    std::deque<StereoVideoWriteJob> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::exception_ptr worker_error_;
    std::thread worker_;
    std::array<cv::VideoWriter, 2> writers_;
    std::array<cv::Mat, 2> blank_frames_;
};

void WriteVideoManifest(
    const std::filesystem::path& output_root,
    const DemoOptions& options,
    const std::array<newnewhand::CameraDescriptor, 2>& active_cameras,
    const std::vector<SavedFrameRecord>& records,
    bool has_calibration_copy) {
    cv::FileStorage fs((output_root / "manifest.yaml").string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open offline capture manifest for writing");
    }

    fs << "format_version" << 2;
    fs << "storage_mode" << "video";
    fs << "capture_fps" << static_cast<int>(options.fps);
    fs << "video_codec" << options.video_codec;
    fs << "video_cam0_path" << "videos/cam0.avi";
    fs << "video_cam1_path" << "videos/cam1.avi";
    fs << "num_frames" << static_cast<int>(records.size());
    fs << "left_camera_serial_number" << active_cameras[0].serial_number;
    fs << "right_camera_serial_number" << active_cameras[1].serial_number;
    fs << "left_camera_model_name" << active_cameras[0].model_name;
    fs << "right_camera_model_name" << active_cameras[1].model_name;
    if (has_calibration_copy) {
        fs << "calibration_file" << "calibration/stereo_calibration.yaml";
    }

    fs << "frames" << "[";
    for (const auto& record : records) {
        fs << "{";
        fs << "capture_index" << static_cast<double>(record.capture_index);
        fs << "trigger_time_us" << record.trigger_time_us;
        fs << "views" << "[";
        for (std::size_t camera_index = 0; camera_index < 2; ++camera_index) {
            fs << "{";
            fs << "camera_index" << static_cast<int>(camera_index);
            fs << "valid" << static_cast<int>(record.valid[camera_index]);
            fs << "frame_index" << static_cast<double>(record.frame_indices[camera_index]);
            fs << "device_timestamp" << static_cast<double>(record.device_timestamps[camera_index]);
            fs << "sdk_host_timestamp" << record.sdk_host_timestamps[camera_index];
            fs << "host_time_us" << record.host_times_us[camera_index];
            fs << "video_frame_index" << record.video_frame_indices[camera_index];
            fs << "}";
        }
        fs << "]";
        fs << "}";
    }
    fs << "]";
}

std::optional<std::filesystem::path> CopyCalibrationIfPresent(
    const std::filesystem::path& input_root,
    const std::filesystem::path& output_root) {
    const std::filesystem::path input_calibration = input_root / "calibration" / "stereo_calibration.yaml";
    if (!std::filesystem::exists(input_calibration)) {
        return std::nullopt;
    }
    const std::filesystem::path output_calibration = output_root / "calibration" / "stereo_calibration.yaml";
    std::filesystem::create_directories(output_calibration.parent_path());
    std::filesystem::copy_file(
        input_calibration,
        output_calibration,
        std::filesystem::copy_options::overwrite_existing);
    return output_calibration;
}

std::vector<SavedFrameRecord> LoadVideoManifest(
    const std::filesystem::path& input_root,
    std::string& video_cam0_path,
    std::string& video_cam1_path) {
    cv::FileStorage fs((input_root / "manifest.yaml").string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open input manifest: " + (input_root / "manifest.yaml").string());
    }

    std::string storage_mode;
    fs["storage_mode"] >> storage_mode;
    if (storage_mode != "video") {
        throw std::runtime_error("input manifest is not a video package");
    }
    fs["video_cam0_path"] >> video_cam0_path;
    fs["video_cam1_path"] >> video_cam1_path;

    std::vector<SavedFrameRecord> records;
    const cv::FileNode frames = fs["frames"];
    for (const auto& frame_node : frames) {
        SavedFrameRecord record;
        std::int64_t capture_index = 0;
        frame_node["capture_index"] >> capture_index;
        record.capture_index = static_cast<std::uint64_t>(capture_index);
        frame_node["trigger_time_us"] >> record.trigger_time_us;
        const cv::FileNode views = frame_node["views"];
        for (const auto& view_node : views) {
            int camera_index = 0;
            view_node["camera_index"] >> camera_index;
            if (camera_index < 0 || camera_index >= 2) {
                continue;
            }
            int valid = 0;
            view_node["valid"] >> valid;
            record.valid[static_cast<std::size_t>(camera_index)] = valid != 0;
            std::int64_t frame_index = 0;
            std::int64_t device_timestamp = 0;
            view_node["frame_index"] >> frame_index;
            view_node["device_timestamp"] >> device_timestamp;
            record.frame_indices[static_cast<std::size_t>(camera_index)] = static_cast<std::uint64_t>(frame_index);
            record.device_timestamps[static_cast<std::size_t>(camera_index)] = static_cast<std::uint64_t>(device_timestamp);
            view_node["sdk_host_timestamp"] >> record.sdk_host_timestamps[static_cast<std::size_t>(camera_index)];
            view_node["host_time_us"] >> record.host_times_us[static_cast<std::size_t>(camera_index)];
            if (!view_node["video_frame_index"].empty()) {
                view_node["video_frame_index"] >> record.video_frame_indices[static_cast<std::size_t>(camera_index)];
            }
        }
        records.push_back(record);
    }
    return records;
}

void WriteImageManifest(
    const std::filesystem::path& output_root,
    const DemoOptions& options,
    const std::vector<SavedFrameRecord>& records,
    bool has_calibration_copy) {
    cv::FileStorage fs((output_root / "manifest.yaml").string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open decoded manifest for writing");
    }

    fs << "format_version" << 2;
    fs << "storage_mode" << "image_sequence";
    fs << "image_format" << options.image_format;
    fs << "num_frames" << static_cast<int>(records.size());
    if (has_calibration_copy) {
        fs << "calibration_file" << "calibration/stereo_calibration.yaml";
    }

    fs << "frames" << "[";
    for (const auto& record : records) {
        fs << "{";
        fs << "capture_index" << static_cast<double>(record.capture_index);
        fs << "trigger_time_us" << record.trigger_time_us;
        fs << "views" << "[";
        for (std::size_t camera_index = 0; camera_index < 2; ++camera_index) {
            std::ostringstream image_name;
            image_name << "images/cam" << camera_index << "/"
                       << std::setw(6) << std::setfill('0') << record.capture_index
                       << "." << options.image_format;
            fs << "{";
            fs << "camera_index" << static_cast<int>(camera_index);
            fs << "valid" << static_cast<int>(record.valid[camera_index]);
            fs << "frame_index" << static_cast<double>(record.frame_indices[camera_index]);
            fs << "device_timestamp" << static_cast<double>(record.device_timestamps[camera_index]);
            fs << "sdk_host_timestamp" << record.sdk_host_timestamps[camera_index];
            fs << "host_time_us" << record.host_times_us[camera_index];
            if (record.valid[camera_index]) {
                fs << "image_path" << image_name.str();
            }
            fs << "}";
        }
        fs << "]";
        fs << "}";
    }
    fs << "]";
}

int ParseFourcc(const std::string& codec) {
    return cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
}

void RunCaptureMode(const DemoOptions& options) {
    newnewhand::StereoCaptureConfig capture_config;
    capture_config.camera_settings.exposure_us = options.exposure_us;
    capture_config.camera_settings.gain = options.gain;
    capture_config.serial_numbers = {options.cam0_serial, options.cam1_serial};

    std::optional<newnewhand::StereoCalibrationResult> calibration;
    if (!options.calibration_path.empty()) {
        calibration = newnewhand::StereoCalibrator::LoadResult(options.calibration_path);
        ApplyCalibrationSerials(options, *calibration, capture_config);
    }

    newnewhand::StereoCapture capture(capture_config);
    capture.Initialize();
    capture.Start();
    const auto active_cameras = capture.ActiveCameras();

    std::filesystem::create_directories(options.output_dir / "videos");
    bool has_calibration_copy = false;
    if (calibration.has_value()) {
        const std::filesystem::path calibration_dir = options.output_dir / "calibration";
        std::filesystem::create_directories(calibration_dir);
        newnewhand::StereoCalibrator::SaveLoadedResult(
            *calibration,
            calibration_dir / "stereo_calibration.yaml");
        has_calibration_copy = true;
    }

    AsyncStereoVideoWriter video_writer(
        options.output_dir,
        ParseFourcc(options.video_codec),
        static_cast<double>(options.fps),
        options.writer_queue_size);

    SharedPreviewState preview_state;
    std::thread preview_worker;
    if (options.preview) {
        preview_worker = std::thread(PreviewWorker, active_cameras, std::ref(preview_state));
    }

    std::vector<SavedFrameRecord> records;
    const auto frame_interval = std::chrono::microseconds(1000000 / options.fps);
    const auto start_time = std::chrono::steady_clock::now();
    double capture_fps = 0.0;
    auto previous_capture_time = std::chrono::steady_clock::time_point{};

    for (int frame_count = 0; options.frames < 0 || frame_count < options.frames; ++frame_count) {
        const auto loop_start = std::chrono::steady_clock::now();
        const auto stereo_frame = capture.Capture();
        if (previous_capture_time != std::chrono::steady_clock::time_point{}) {
            const double dt_seconds =
                std::chrono::duration<double>(stereo_frame.trigger_timestamp - previous_capture_time).count();
            if (dt_seconds > 1e-6) {
                const double instant_fps = 1.0 / dt_seconds;
                capture_fps = capture_fps <= 0.0
                    ? instant_fps
                    : 0.85 * capture_fps + 0.15 * instant_fps;
            }
        }
        previous_capture_time = stereo_frame.trigger_timestamp;

        SavedFrameRecord record;
        record.capture_index = stereo_frame.capture_index;
        record.trigger_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            stereo_frame.trigger_timestamp - start_time).count();
        StereoVideoWriteJob video_job;

        for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
            const auto& frame = stereo_frame.views[camera_index];
            record.valid[camera_index] = frame.valid && !frame.bgr_image.empty();
            record.frame_indices[camera_index] = frame.frame_index;
            record.device_timestamps[camera_index] = frame.device_timestamp;
            record.sdk_host_timestamps[camera_index] = frame.sdk_host_timestamp;
            record.host_times_us[camera_index] = std::chrono::duration_cast<std::chrono::microseconds>(
                frame.host_timestamp - start_time).count();
            record.video_frame_indices[camera_index] = static_cast<std::int64_t>(records.size());

            if (!frame.valid || frame.bgr_image.empty()) {
                std::cerr
                    << "[offline-capture] capture=" << stereo_frame.capture_index
                    << " cam" << camera_index
                    << " failed: " << frame.error_message
                    << "\n";
                continue;
            }

            video_job.valid[camera_index] = true;
            video_job.images[camera_index] = frame.bgr_image.clone();
        }

        video_writer.Enqueue(std::move(video_job));
        records.push_back(record);

        if (options.preview) {
            auto preview_frame = std::make_shared<PreviewFrame>();
            preview_frame->capture_index = stereo_frame.capture_index;
            preview_frame->capture_fps = capture_fps > 0.0 ? capture_fps : static_cast<double>(options.fps);
            for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
                const auto& frame = stereo_frame.views[camera_index];
                preview_frame->valid[camera_index] = frame.valid && !frame.bgr_image.empty();
                if (preview_frame->valid[camera_index]) {
                    preview_frame->images[camera_index] = frame.bgr_image.clone();
                }
            }
            {
                std::lock_guard<std::mutex> lock(preview_state.mutex);
                preview_state.latest_frame = std::move(preview_frame);
                if (preview_state.quit_requested) {
                    break;
                }
            }
            preview_state.cv.notify_one();
        }

        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        const auto remaining = frame_interval - std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        if (remaining.count() > 0) {
            std::this_thread::sleep_for(remaining);
        }
    }

    capture.Stop();
    video_writer.Close();
    if (options.preview) {
        {
            std::lock_guard<std::mutex> lock(preview_state.mutex);
            preview_state.stop_requested = true;
            preview_state.latest_frame.reset();
        }
        preview_state.cv.notify_all();
        if (preview_worker.joinable()) {
            preview_worker.join();
        }
    }
    WriteVideoManifest(options.output_dir, options, active_cameras, records, has_calibration_copy);
}

void RunDecodeMode(const DemoOptions& options) {
    std::string video_cam0_path;
    std::string video_cam1_path;
    auto records = LoadVideoManifest(options.input_dir, video_cam0_path, video_cam1_path);
    if (records.empty()) {
        throw std::runtime_error("input video package does not contain any frames");
    }
    if (options.frames > 0 && static_cast<std::size_t>(options.frames) < records.size()) {
        records.resize(static_cast<std::size_t>(options.frames));
    }

    cv::VideoCapture capture0((options.input_dir / video_cam0_path).string());
    cv::VideoCapture capture1((options.input_dir / video_cam1_path).string());
    if (!capture0.isOpened()) {
        throw std::runtime_error("failed to open video: " + (options.input_dir / video_cam0_path).string());
    }
    if (!capture1.isOpened()) {
        throw std::runtime_error("failed to open video: " + (options.input_dir / video_cam1_path).string());
    }

    std::filesystem::create_directories(options.output_dir / "images" / "cam0");
    std::filesystem::create_directories(options.output_dir / "images" / "cam1");
    const bool has_calibration_copy = CopyCalibrationIfPresent(options.input_dir, options.output_dir).has_value();

    for (const auto& record : records) {
        cv::Mat frame0;
        cv::Mat frame1;
        if (!capture0.read(frame0)) {
            throw std::runtime_error("failed to decode frame from cam0 video");
        }
        if (!capture1.read(frame1)) {
            throw std::runtime_error("failed to decode frame from cam1 video");
        }

        if (record.valid[0]) {
            const auto output_path = BuildFrameImagePath(options.output_dir, 0, record.capture_index, options.image_format);
            std::filesystem::create_directories(output_path.parent_path());
            if (!cv::imwrite(output_path.string(), frame0)) {
                throw std::runtime_error("failed to write decoded image: " + output_path.string());
            }
        }
        if (record.valid[1]) {
            const auto output_path = BuildFrameImagePath(options.output_dir, 1, record.capture_index, options.image_format);
            std::filesystem::create_directories(output_path.parent_path());
            if (!cv::imwrite(output_path.string(), frame1)) {
                throw std::runtime_error("failed to write decoded image: " + output_path.string());
            }
        }

        if (options.preview) {
            cv::Mat preview0;
            cv::Mat preview1;
            cv::resize(frame0, preview0, cv::Size(), 0.5, 0.5);
            cv::resize(frame1, preview1, cv::Size(), 0.5, 0.5);
            cv::imshow("offline_decode_cam0", preview0);
            cv::imshow("offline_decode_cam1", preview1);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }
    }

    WriteImageManifest(options.output_dir, options, records, has_calibration_copy);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const DemoOptions options = ParseArgs(argc, argv);
        if (options.mode == "capture") {
            RunCaptureMode(options);
        } else {
            RunDecodeMode(options);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
