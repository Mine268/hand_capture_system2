#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <opencv2/core/mat.hpp>

namespace newnewhand {

struct CameraDescriptor {
    std::size_t device_index = 0;
    std::string model_name;
    std::string serial_number;
};

struct CameraFrame {
    std::size_t camera_index = 0;
    std::string serial_number;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frame_index = 0;
    std::uint64_t device_timestamp = 0;
    std::int64_t sdk_host_timestamp = 0;
    std::chrono::steady_clock::time_point host_timestamp;
    bool valid = false;
    std::string error_message;
    cv::Mat bgr_image;

    bool empty() const {
        return !valid || bgr_image.empty();
    }
};

struct StereoFrame {
    std::uint64_t capture_index = 0;
    std::chrono::steady_clock::time_point trigger_timestamp;
    std::array<CameraFrame, 2> views;

    bool is_complete() const {
        return !views[0].empty() && !views[1].empty();
    }
};

}  // namespace newnewhand
