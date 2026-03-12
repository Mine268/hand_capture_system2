#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "newnewhand/types/stereo_frame.h"

namespace newnewhand {

struct StereoCameraSettings {
    float exposure_us = 10000.0f;
    float gain = -1.0f;
    int trigger_timeout_ms = 1000;
    bool enable_gamma = true;
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace newnewhand
