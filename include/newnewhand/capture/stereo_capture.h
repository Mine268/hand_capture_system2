#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "newnewhand/types/stereo_frame.h"

namespace newnewhand {

enum class StereoPixelFormatMode {
    kRawPreferred,
    kMono8,
    kBgr8,
    kRgb8,
};

struct StereoCaptureRuntimeInfo {
    std::string serial_number;
    std::string pixel_format;
    std::uint64_t payload_size = 0;
};

struct StereoCaptureTimingCameraInfo {
    std::string pixel_format;
    std::uint64_t frame_bytes = 0;
    double wait_frame_ms = 0.0;
    double image_process_ms = 0.0;
    double total_worker_ms = 0.0;
};

struct StereoCaptureTimingInfo {
    std::uint64_t capture_index = 0;
    double trigger_ms = 0.0;
    double assemble_ms = 0.0;
    double total_capture_ms = 0.0;
    std::array<StereoCaptureTimingCameraInfo, 2> cameras;
};

struct StereoCameraSettings {
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    int trigger_timeout_ms = 1000;
    bool enable_gamma = true;
    bool include_image_data = true;
    StereoPixelFormatMode pixel_format = StereoPixelFormatMode::kRawPreferred;
};

struct StereoCaptureConfig {
    std::array<std::string, 2> serial_numbers;
    StereoCameraSettings camera_settings;
};

class StereoCapture {
public:
    explicit StereoCapture(StereoCaptureConfig config = {});
    ~StereoCapture();

    StereoCapture(const StereoCapture&) = delete;
    StereoCapture& operator=(const StereoCapture&) = delete;

    StereoCapture(StereoCapture&&) noexcept;
    StereoCapture& operator=(StereoCapture&&) noexcept;

    static std::vector<CameraDescriptor> EnumerateConnectedCameras();

    void Initialize();
    void Start();
    StereoFrame Capture();
    void Stop();
    void Shutdown();

    bool IsInitialized() const;
    bool IsRunning() const;
    std::array<CameraDescriptor, 2> ActiveCameras() const;
    std::array<StereoCaptureRuntimeInfo, 2> RuntimeInfo() const;
    StereoCaptureTimingInfo LastCaptureTiming() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace newnewhand
