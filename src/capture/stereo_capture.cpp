#include "newnewhand/capture/stereo_capture.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "mv_camera.h"

namespace newnewhand {

namespace {

constexpr unsigned int kSupportedTransportLayers = MV_USB_DEVICE | MV_GIGE_DEVICE;
constexpr std::size_t kStereoCameraCount = 2;

std::string TrimNullTerminatedString(const unsigned char* raw) {
    if (!raw) {
        return {};
    }

    return std::string(reinterpret_cast<const char*>(raw));
}

std::string ExtractSerialNumber(const MV_CC_DEVICE_INFO& info) {
    if (info.nTLayerType == MV_USB_DEVICE) {
        return TrimNullTerminatedString(info.SpecialInfo.stUsb3VInfo.chSerialNumber);
    }
    if (info.nTLayerType == MV_GIGE_DEVICE) {
        return TrimNullTerminatedString(info.SpecialInfo.stGigEInfo.chSerialNumber);
    }
    return {};
}

std::string ExtractModelName(const MV_CC_DEVICE_INFO& info) {
    if (info.nTLayerType == MV_USB_DEVICE) {
        return TrimNullTerminatedString(info.SpecialInfo.stUsb3VInfo.chModelName);
    }
    if (info.nTLayerType == MV_GIGE_DEVICE) {
        return TrimNullTerminatedString(info.SpecialInfo.stGigEInfo.chModelName);
    }
    return {};
}

std::string FormatMvError(const std::string& action, int code) {
    std::ostringstream oss;
    oss << action << " failed with MVS error 0x" << std::hex << code;
    return oss.str();
}

void ThrowIfMvFailed(const std::string& action, int code) {
    if (code != MV_OK) {
        throw std::runtime_error(FormatMvError(action, code));
    }
}

void TrySetEnumValue(CMvCamera& camera, const char* key, unsigned int value) {
    camera.SetEnumValue(key, value);
}

void TrySetBoolValue(CMvCamera& camera, const char* key, bool value) {
    camera.SetBoolValue(key, value);
}

std::vector<CameraDescriptor> EnumerateCameraDescriptors() {
    MV_CC_DEVICE_INFO_LIST device_info_list = {0};
    ThrowIfMvFailed("enumerate devices", CMvCamera::EnumDevices(kSupportedTransportLayers, &device_info_list));

    std::vector<CameraDescriptor> descriptors;
    descriptors.reserve(device_info_list.nDeviceNum);
    for (unsigned int device_index = 0; device_index < device_info_list.nDeviceNum; ++device_index) {
        const MV_CC_DEVICE_INFO* info = device_info_list.pDeviceInfo[device_index];
        if (!info) {
            continue;
        }

        CameraDescriptor descriptor;
        descriptor.device_index = device_index;
        descriptor.model_name = ExtractModelName(*info);
        descriptor.serial_number = ExtractSerialNumber(*info);
        descriptors.push_back(std::move(descriptor));
    }
    return descriptors;
}

struct CaptureResult {
    bool valid = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frame_index = 0;
    std::uint64_t device_timestamp = 0;
    std::int64_t sdk_host_timestamp = 0;
    std::chrono::steady_clock::time_point host_timestamp;
    std::string error_message;
    cv::Mat bgr_image;
};

struct DeviceContext {
    CameraDescriptor descriptor;
    std::unique_ptr<CMvCamera> camera;
    std::vector<unsigned char> raw_buffer;
    std::thread worker_thread;
    std::mutex mutex;
    std::condition_variable trigger_cv;
    std::condition_variable completion_cv;
    std::uint64_t requested_capture_seq = 0;
    std::uint64_t completed_capture_seq = 0;
    bool shutdown_requested = false;
    CaptureResult last_result;
};

void ConfigureCamera(CMvCamera& camera, const StereoCameraSettings& settings) {
    ThrowIfMvFailed("set TriggerMode", camera.SetEnumValue("TriggerMode", 1));
    ThrowIfMvFailed("set TriggerSource", camera.SetEnumValue("TriggerSource", 7));
    ThrowIfMvFailed("set PixelFormat", camera.SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed));

    if (settings.enable_gamma) {
        TrySetBoolValue(camera, "GammaEnable", true);
        TrySetEnumValue(camera, "GammaSelector", 2);
    } else {
        TrySetBoolValue(camera, "GammaEnable", false);
    }

    ThrowIfMvFailed("set ExposureMode", camera.SetEnumValue("ExposureMode", 0));
    ThrowIfMvFailed("set ExposureTime", camera.SetFloatValue("ExposureTime", settings.exposure_us));

    if (settings.gain < 0.0f) {
        ThrowIfMvFailed("set GainAuto", camera.SetEnumValue("GainAuto", 1));
    } else {
        ThrowIfMvFailed("set GainAuto", camera.SetEnumValue("GainAuto", 0));
        ThrowIfMvFailed("set Gain", camera.SetFloatValue("Gain", settings.gain));
    }
}

std::array<CameraDescriptor, 2> SelectStereoDescriptors(
    const std::vector<CameraDescriptor>& available,
    const std::array<std::string, 2>& serial_numbers) {
    if (available.size() < kStereoCameraCount) {
        throw std::runtime_error("at least two cameras are required for stereo capture");
    }

    std::array<CameraDescriptor, 2> selected;
    std::array<bool, 2> selected_slot = {false, false};
    std::vector<bool> used(available.size(), false);

    for (std::size_t slot = 0; slot < kStereoCameraCount; ++slot) {
        const std::string& requested_serial = serial_numbers[slot];
        if (requested_serial.empty()) {
            continue;
        }

        auto it = std::find_if(
            available.begin(),
            available.end(),
            [&](const CameraDescriptor& descriptor) {
                return descriptor.serial_number == requested_serial;
            });
        if (it == available.end()) {
            throw std::runtime_error("requested camera serial not found: " + requested_serial);
        }

        const std::size_t index = static_cast<std::size_t>(std::distance(available.begin(), it));
        if (used[index]) {
            throw std::runtime_error("duplicate serial requested for stereo pair: " + requested_serial);
        }

        used[index] = true;
        selected[slot] = *it;
        selected_slot[slot] = true;
    }

    for (std::size_t slot = 0; slot < kStereoCameraCount; ++slot) {
        if (selected_slot[slot]) {
            continue;
        }

        const auto it = std::find(used.begin(), used.end(), false);
        if (it == used.end()) {
            break;
        }

        const std::size_t index = static_cast<std::size_t>(std::distance(used.begin(), it));
        used[index] = true;
        selected[slot] = available[index];
        selected_slot[slot] = true;
    }

    if (!selected_slot[0] || !selected_slot[1]) {
        throw std::runtime_error("failed to select two unique cameras for stereo capture");
    }

    return selected;
}

}  // namespace

struct StereoCapture::Impl {
    explicit Impl(StereoCaptureConfig config_in) : config(std::move(config_in)) {}

    StereoCaptureConfig config;
    bool initialized = false;
    bool running = false;
    std::uint64_t capture_counter = 0;
    std::array<CameraDescriptor, 2> active_cameras;
    std::array<std::unique_ptr<DeviceContext>, 2> devices;

    void Initialize() {
        if (initialized) {
            return;
        }

        const std::vector<CameraDescriptor> available = EnumerateCameraDescriptors();
        const std::array<CameraDescriptor, 2> selected = SelectStereoDescriptors(available, config.serial_numbers);

        for (auto& device : devices) {
            device.reset();
        }

        MV_CC_DEVICE_INFO_LIST device_info_list = {0};
        ThrowIfMvFailed("enumerate devices", CMvCamera::EnumDevices(kSupportedTransportLayers, &device_info_list));

        for (std::size_t camera_index = 0; camera_index < kStereoCameraCount; ++camera_index) {
            const CameraDescriptor& descriptor = selected[camera_index];
            auto* device_info = device_info_list.pDeviceInfo[descriptor.device_index];
            if (!device_info) {
                throw std::runtime_error("camera device info missing for index " + std::to_string(descriptor.device_index));
            }

            auto context = std::make_unique<DeviceContext>();
            context->descriptor = descriptor;
            context->camera = std::make_unique<CMvCamera>();
            ThrowIfMvFailed(
                "open camera " + descriptor.serial_number,
                context->camera->Open(device_info));
            ConfigureCamera(*context->camera, config.camera_settings);

            MVCC_INTVALUE_EX payload_size = {0};
            ThrowIfMvFailed(
                "query PayloadSize for " + descriptor.serial_number,
                context->camera->GetIntValue("PayloadSize", &payload_size));
            context->raw_buffer.resize(static_cast<std::size_t>(payload_size.nCurValue));

            devices[camera_index] = std::move(context);
            active_cameras[camera_index] = descriptor;
        }

        initialized = true;
    }

    void WorkerLoop(DeviceContext* context, std::size_t camera_index) {
        while (true) {
            std::uint64_t target_capture_seq = 0;
            {
                std::unique_lock<std::mutex> lock(context->mutex);
                context->trigger_cv.wait(lock, [&]() {
                    return context->shutdown_requested
                        || context->requested_capture_seq > context->completed_capture_seq;
                });
                if (context->shutdown_requested) {
                    return;
                }
                target_capture_seq = context->requested_capture_seq;
            }

            CaptureResult result;
            result.host_timestamp = std::chrono::steady_clock::now();

            MV_FRAME_OUT_INFO_EX frame_info = {0};
            const int ret = context->camera->GetOneFrameTimeout(
                context->raw_buffer.data(),
                static_cast<unsigned int>(context->raw_buffer.size()),
                &frame_info,
                config.camera_settings.trigger_timeout_ms);
            if (ret == MV_OK) {
                cv::Mat rgb(
                    static_cast<int>(frame_info.nHeight),
                    static_cast<int>(frame_info.nWidth),
                    CV_8UC3,
                    context->raw_buffer.data());
                cv::cvtColor(rgb, result.bgr_image, cv::COLOR_RGB2BGR);

                result.valid = true;
                result.width = frame_info.nWidth;
                result.height = frame_info.nHeight;
                result.frame_index = frame_info.nFrameNum;
                result.device_timestamp =
                    (static_cast<std::uint64_t>(frame_info.nDevTimeStampHigh) << 32)
                    | frame_info.nDevTimeStampLow;
                result.sdk_host_timestamp = frame_info.nHostTimeStamp;
            } else {
                result.error_message = FormatMvError(
                    "capture frame on camera[" + std::to_string(camera_index) + "]",
                    ret);
            }

            {
                std::lock_guard<std::mutex> lock(context->mutex);
                context->last_result = std::move(result);
                context->completed_capture_seq = target_capture_seq;
            }
            context->completion_cv.notify_all();
        }
    }

    void Start() {
        if (!initialized) {
            Initialize();
        }
        if (running) {
            return;
        }

        capture_counter = 0;
        for (std::size_t camera_index = 0; camera_index < devices.size(); ++camera_index) {
            DeviceContext& context = *devices[camera_index];
            context.shutdown_requested = false;
            context.requested_capture_seq = 0;
            context.completed_capture_seq = 0;
            ThrowIfMvFailed(
                "start grabbing on " + context.descriptor.serial_number,
                context.camera->StartGrabbing());
            context.worker_thread = std::thread([this, &context, camera_index]() {
                WorkerLoop(&context, camera_index);
            });
        }

        running = true;
    }

    StereoFrame Capture() {
        if (!running) {
            throw std::runtime_error("StereoCapture::Capture called before Start");
        }

        const std::uint64_t capture_seq = ++capture_counter;
        StereoFrame stereo_frame;
        stereo_frame.capture_index = capture_seq;

        for (const auto& device : devices) {
            DeviceContext& context = *device;
            std::lock_guard<std::mutex> lock(context.mutex);
            context.requested_capture_seq = capture_seq;
        }
        for (const auto& device : devices) {
            DeviceContext& context = *device;
            context.trigger_cv.notify_one();
        }

        stereo_frame.trigger_timestamp = std::chrono::steady_clock::now();
        for (const auto& device : devices) {
            DeviceContext& context = *device;
            ThrowIfMvFailed(
                "software trigger on " + context.descriptor.serial_number,
                context.camera->CommandExecute("TriggerSoftware"));
        }

        for (std::size_t camera_index = 0; camera_index < devices.size(); ++camera_index) {
            DeviceContext& context = *devices[camera_index];
            std::unique_lock<std::mutex> lock(context.mutex);
            context.completion_cv.wait(lock, [&]() {
                return context.completed_capture_seq >= capture_seq;
            });

            CameraFrame frame;
            frame.camera_index = camera_index;
            frame.serial_number = context.descriptor.serial_number;
            frame.width = context.last_result.width;
            frame.height = context.last_result.height;
            frame.frame_index = context.last_result.frame_index;
            frame.device_timestamp = context.last_result.device_timestamp;
            frame.sdk_host_timestamp = context.last_result.sdk_host_timestamp;
            frame.host_timestamp = context.last_result.host_timestamp;
            frame.valid = context.last_result.valid;
            frame.error_message = context.last_result.error_message;
            frame.bgr_image = context.last_result.bgr_image.clone();
            stereo_frame.views[camera_index] = std::move(frame);
        }

        return stereo_frame;
    }

    void Stop() {
        if (!running) {
            return;
        }

        for (const auto& device : devices) {
            DeviceContext& context = *device;
            {
                std::lock_guard<std::mutex> lock(context.mutex);
                context.shutdown_requested = true;
            }
            context.trigger_cv.notify_one();
        }

        for (const auto& device : devices) {
            DeviceContext& context = *device;
            context.camera->StopGrabbing();
        }
        for (const auto& device : devices) {
            DeviceContext& context = *device;
            if (context.worker_thread.joinable()) {
                context.worker_thread.join();
            }
        }

        running = false;
    }

    void Shutdown() {
        Stop();
        if (!initialized) {
            return;
        }

        for (auto& device : devices) {
            if (device && device->camera) {
                device->camera->Close();
            }
            device.reset();
        }
        initialized = false;
    }
};

StereoCapture::StereoCapture(StereoCaptureConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

StereoCapture::~StereoCapture() {
    if (impl_) {
        impl_->Shutdown();
    }
}

StereoCapture::StereoCapture(StereoCapture&&) noexcept = default;

StereoCapture& StereoCapture::operator=(StereoCapture&&) noexcept = default;

std::vector<CameraDescriptor> StereoCapture::EnumerateConnectedCameras() {
    return EnumerateCameraDescriptors();
}

void StereoCapture::Initialize() {
    impl_->Initialize();
}

void StereoCapture::Start() {
    impl_->Start();
}

StereoFrame StereoCapture::Capture() {
    return impl_->Capture();
}

void StereoCapture::Stop() {
    impl_->Stop();
}

void StereoCapture::Shutdown() {
    impl_->Shutdown();
}

bool StereoCapture::IsInitialized() const {
    return impl_->initialized;
}

bool StereoCapture::IsRunning() const {
    return impl_->running;
}

std::array<CameraDescriptor, 2> StereoCapture::ActiveCameras() const {
    return impl_->active_cameras;
}

}  // namespace newnewhand
