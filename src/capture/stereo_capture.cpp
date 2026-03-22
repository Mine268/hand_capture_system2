#include "newnewhand/capture/stereo_capture.h"

#include <algorithm>
#include <array>
#include <climits>
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
constexpr unsigned int kDefaultImageNodeNum = 3;

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

void TrySetIntValue(CMvCamera& camera, const char* key, std::int64_t value) {
    camera.SetIntValue(key, value);
}

void SetFloatValueChecked(CMvCamera& camera, const char* key, float value) {
    MVCC_FLOATVALUE range = {0};
    const int get_ret = camera.GetFloatValue(key, &range);
    if (get_ret == MV_OK) {
        if (value < range.fMin || value > range.fMax) {
            std::ostringstream oss;
            oss
                << key << " value out of range: requested=" << value
                << " supported=[" << range.fMin << ", " << range.fMax << "]";
            throw std::runtime_error(oss.str());
        }
    } else if (get_ret != MV_E_SUPPORT) {
        ThrowIfMvFailed(std::string("query ") + key + " range", get_ret);
    }

    ThrowIfMvFailed(std::string("set ") + key, camera.SetFloatValue(key, value));
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

std::uint32_t FrameWidth(const MV_FRAME_OUT_INFO_EX& frame_info) {
    return frame_info.nExtendWidth != 0 ? frame_info.nExtendWidth : frame_info.nWidth;
}

std::uint32_t FrameHeight(const MV_FRAME_OUT_INFO_EX& frame_info) {
    return frame_info.nExtendHeight != 0 ? frame_info.nExtendHeight : frame_info.nHeight;
}

std::uint64_t FrameLengthBytes(const MV_FRAME_OUT_INFO_EX& frame_info) {
    return frame_info.nFrameLenEx != 0 ? frame_info.nFrameLenEx : frame_info.nFrameLen;
}

std::string PixelTypeToString(MvGvspPixelType pixel_type) {
    switch (pixel_type) {
        case PixelType_Gvsp_Mono8:
            return "Mono8";
        case PixelType_Gvsp_Mono10:
            return "Mono10";
        case PixelType_Gvsp_Mono10_Packed:
            return "Mono10_Packed";
        case PixelType_Gvsp_Mono12:
            return "Mono12";
        case PixelType_Gvsp_Mono12_Packed:
            return "Mono12_Packed";
        case PixelType_Gvsp_BayerGR8:
            return "BayerGR8";
        case PixelType_Gvsp_BayerRG8:
            return "BayerRG8";
        case PixelType_Gvsp_BayerGB8:
            return "BayerGB8";
        case PixelType_Gvsp_BayerBG8:
            return "BayerBG8";
        case PixelType_Gvsp_BayerRBGG8:
            return "BayerRBGG8";
        case PixelType_Gvsp_BayerGR10:
            return "BayerGR10";
        case PixelType_Gvsp_BayerRG10:
            return "BayerRG10";
        case PixelType_Gvsp_BayerGB10:
            return "BayerGB10";
        case PixelType_Gvsp_BayerBG10:
            return "BayerBG10";
        case PixelType_Gvsp_BayerGR10_Packed:
            return "BayerGR10_Packed";
        case PixelType_Gvsp_BayerRG10_Packed:
            return "BayerRG10_Packed";
        case PixelType_Gvsp_BayerGB10_Packed:
            return "BayerGB10_Packed";
        case PixelType_Gvsp_BayerBG10_Packed:
            return "BayerBG10_Packed";
        case PixelType_Gvsp_BayerGR12:
            return "BayerGR12";
        case PixelType_Gvsp_BayerRG12:
            return "BayerRG12";
        case PixelType_Gvsp_BayerGB12:
            return "BayerGB12";
        case PixelType_Gvsp_BayerBG12:
            return "BayerBG12";
        case PixelType_Gvsp_BayerGR12_Packed:
            return "BayerGR12_Packed";
        case PixelType_Gvsp_BayerRG12_Packed:
            return "BayerRG12_Packed";
        case PixelType_Gvsp_BayerGB12_Packed:
            return "BayerGB12_Packed";
        case PixelType_Gvsp_BayerBG12_Packed:
            return "BayerBG12_Packed";
        case PixelType_Gvsp_RGB8_Packed:
            return "RGB8_Packed";
        case PixelType_Gvsp_BGR8_Packed:
            return "BGR8_Packed";
        default: {
            std::ostringstream oss;
            oss << "0x" << std::hex << static_cast<unsigned int>(pixel_type);
            return oss.str();
        }
    }
}

std::string EnumEntrySymbolic(CMvCamera& camera, const char* key, unsigned int value) {
    MVCC_ENUMENTRY entry = {0};
    entry.nValue = value;
    const int ret = camera.GetEnumEntrySymbolic(key, &entry);
    if (ret != MV_OK) {
        return {};
    }
    return entry.chSymbolic;
}

bool EnumSupportsValue(const MVCC_ENUMVALUE& enum_value, unsigned int value) {
    return std::find(
        enum_value.nSupportValue,
        enum_value.nSupportValue + enum_value.nSupportedNum,
        value) != (enum_value.nSupportValue + enum_value.nSupportedNum);
}

std::vector<unsigned int> PixelFormatCandidates(StereoPixelFormatMode mode) {
    switch (mode) {
        case StereoPixelFormatMode::kMono8:
            return {PixelType_Gvsp_Mono8};
        case StereoPixelFormatMode::kBgr8:
            return {PixelType_Gvsp_BGR8_Packed};
        case StereoPixelFormatMode::kRgb8:
            return {PixelType_Gvsp_RGB8_Packed};
        case StereoPixelFormatMode::kRawPreferred:
            return {
                PixelType_Gvsp_BayerGR8,
                PixelType_Gvsp_BayerRG8,
                PixelType_Gvsp_BayerGB8,
                PixelType_Gvsp_BayerBG8,
                PixelType_Gvsp_BayerRBGG8,
                PixelType_Gvsp_BayerGR10_Packed,
                PixelType_Gvsp_BayerRG10_Packed,
                PixelType_Gvsp_BayerGB10_Packed,
                PixelType_Gvsp_BayerBG10_Packed,
                PixelType_Gvsp_BayerGR10,
                PixelType_Gvsp_BayerRG10,
                PixelType_Gvsp_BayerGB10,
                PixelType_Gvsp_BayerBG10,
                PixelType_Gvsp_BayerGR12_Packed,
                PixelType_Gvsp_BayerRG12_Packed,
                PixelType_Gvsp_BayerGB12_Packed,
                PixelType_Gvsp_BayerBG12_Packed,
                PixelType_Gvsp_BayerGR12,
                PixelType_Gvsp_BayerRG12,
                PixelType_Gvsp_BayerGB12,
                PixelType_Gvsp_BayerBG12,
                PixelType_Gvsp_Mono8,
                PixelType_Gvsp_Mono10_Packed,
                PixelType_Gvsp_Mono10,
                PixelType_Gvsp_Mono12_Packed,
                PixelType_Gvsp_Mono12,
                PixelType_Gvsp_BGR8_Packed,
                PixelType_Gvsp_RGB8_Packed,
            };
    }

    return {};
}

std::string PixelFormatModeToString(StereoPixelFormatMode mode) {
    switch (mode) {
        case StereoPixelFormatMode::kRawPreferred:
            return "raw";
        case StereoPixelFormatMode::kMono8:
            return "mono8";
        case StereoPixelFormatMode::kBgr8:
            return "bgr8";
        case StereoPixelFormatMode::kRgb8:
            return "rgb8";
    }

    return "unknown";
}

unsigned int SelectPixelFormat(CMvCamera& camera, StereoPixelFormatMode mode) {
    MVCC_ENUMVALUE pixel_formats = {0};
    ThrowIfMvFailed("query PixelFormat", camera.GetEnumValue("PixelFormat", &pixel_formats));

    const auto candidates = PixelFormatCandidates(mode);
    for (const unsigned int candidate : candidates) {
        if (EnumSupportsValue(pixel_formats, candidate)) {
            return candidate;
        }
    }

    if (mode == StereoPixelFormatMode::kRawPreferred && pixel_formats.nCurValue != 0) {
        return pixel_formats.nCurValue;
    }

    std::ostringstream oss;
    oss << "requested pixel format mode '" << PixelFormatModeToString(mode) << "' is not supported";
    throw std::runtime_error(oss.str());
}

std::uint32_t CheckedBgrBufferSize(std::uint32_t width, std::uint32_t height) {
    const std::uint64_t bytes = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 3ULL;
    if (bytes > static_cast<std::uint64_t>(UINT_MAX)) {
        throw std::runtime_error("BGR destination buffer exceeds SDK conversion size limit");
    }
    return static_cast<std::uint32_t>(bytes);
}

std::uint32_t CheckedFrameLength(const MV_FRAME_OUT_INFO_EX& frame_info) {
    const std::uint64_t bytes = FrameLengthBytes(frame_info);
    if (bytes > static_cast<std::uint64_t>(UINT_MAX)) {
        throw std::runtime_error("source frame exceeds SDK conversion size limit");
    }
    return static_cast<std::uint32_t>(bytes);
}

void ConvertFrameToBgr(CMvCamera& camera, const MV_FRAME_OUT& frame, cv::Mat& dst_image) {
    const std::uint32_t width = FrameWidth(frame.stFrameInfo);
    const std::uint32_t height = FrameHeight(frame.stFrameInfo);
    if (width == 0 || height == 0) {
        throw std::runtime_error("invalid frame dimensions");
    }

    if (frame.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed) {
        dst_image = cv::Mat(
            static_cast<int>(height),
            static_cast<int>(width),
            CV_8UC3,
            frame.pBufAddr).clone();
        return;
    }

    dst_image.create(static_cast<int>(height), static_cast<int>(width), CV_8UC3);

    MV_CC_PIXEL_CONVERT_PARAM_EX convert_param = {0};
    convert_param.nWidth = width;
    convert_param.nHeight = height;
    convert_param.enSrcPixelType = frame.stFrameInfo.enPixelType;
    convert_param.pSrcData = frame.pBufAddr;
    convert_param.nSrcDataLen = CheckedFrameLength(frame.stFrameInfo);
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    convert_param.pDstBuffer = dst_image.data;
    convert_param.nDstBufferSize = CheckedBgrBufferSize(width, height);

    ThrowIfMvFailed("convert frame to BGR8", camera.ConvertPixelTypeEx(&convert_param));
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
    std::string pixel_format;
    std::uint64_t frame_bytes = 0;
    double wait_frame_ms = 0.0;
    double image_process_ms = 0.0;
    double total_worker_ms = 0.0;
    cv::Mat bgr_image;
};

struct DeviceContext {
    CameraDescriptor descriptor;
    StereoCaptureRuntimeInfo runtime_info;
    std::unique_ptr<CMvCamera> camera;
    std::thread worker_thread;
    std::mutex mutex;
    std::condition_variable trigger_cv;
    std::condition_variable completion_cv;
    std::uint64_t requested_capture_seq = 0;
    std::uint64_t completed_capture_seq = 0;
    bool shutdown_requested = false;
    CaptureResult last_result;
};

void TuneTransport(CMvCamera& camera, const MV_CC_DEVICE_INFO& device_info) {
    camera.SetImageNodeNum(kDefaultImageNodeNum);
    if (device_info.nTLayerType != MV_GIGE_DEVICE) {
        return;
    }

    unsigned int packet_size = 0;
    if (camera.GetOptimalPacketSize(&packet_size) == MV_OK && packet_size > 0) {
        TrySetIntValue(camera, "GevSCPSPacketSize", packet_size);
    }
}

void ConfigureCamera(
    CMvCamera& camera,
    const MV_CC_DEVICE_INFO& device_info,
    const StereoCameraSettings& settings,
    StereoCaptureRuntimeInfo* runtime_info) {
    TuneTransport(camera, device_info);
    TrySetBoolValue(camera, "AcquisitionFrameRateEnable", false);
    ThrowIfMvFailed("set TriggerMode", camera.SetEnumValue("TriggerMode", 1));
    ThrowIfMvFailed("set TriggerSource", camera.SetEnumValue("TriggerSource", 7));

    const unsigned int pixel_format = SelectPixelFormat(camera, settings.pixel_format);
    ThrowIfMvFailed("set PixelFormat", camera.SetEnumValue("PixelFormat", pixel_format));
    if (runtime_info) {
        const std::string symbolic = EnumEntrySymbolic(camera, "PixelFormat", pixel_format);
        runtime_info->pixel_format = symbolic.empty() ? PixelTypeToString(static_cast<MvGvspPixelType>(pixel_format)) : symbolic;
    }

    if (settings.enable_gamma) {
        TrySetBoolValue(camera, "GammaEnable", true);
        TrySetEnumValue(camera, "GammaSelector", 2);
    } else {
        TrySetBoolValue(camera, "GammaEnable", false);
    }

    ThrowIfMvFailed("set ExposureMode", camera.SetEnumValue("ExposureMode", 0));
    SetFloatValueChecked(camera, "ExposureTime", settings.exposure_us);

    if (settings.gain < 0.0f) {
        ThrowIfMvFailed("set GainAuto", camera.SetEnumValue("GainAuto", 1));
    } else {
        ThrowIfMvFailed("set GainAuto", camera.SetEnumValue("GainAuto", 0));
        SetFloatValueChecked(camera, "Gain", settings.gain);
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
    std::array<StereoCaptureRuntimeInfo, 2> runtime_info;
    StereoCaptureTimingInfo last_capture_timing;
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
            context->runtime_info.serial_number = descriptor.serial_number;
            context->camera = std::make_unique<CMvCamera>();
            ThrowIfMvFailed(
                "open camera " + descriptor.serial_number,
                context->camera->Open(device_info));
            ConfigureCamera(*context->camera, *device_info, config.camera_settings, &context->runtime_info);

            MVCC_INTVALUE_EX payload_size = {0};
            ThrowIfMvFailed(
                "query PayloadSize for " + descriptor.serial_number,
                context->camera->GetIntValue("PayloadSize", &payload_size));
            context->runtime_info.payload_size = static_cast<std::uint64_t>(payload_size.nCurValue);

            devices[camera_index] = std::move(context);
            active_cameras[camera_index] = descriptor;
            runtime_info[camera_index] = devices[camera_index]->runtime_info;
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

            const auto worker_start = std::chrono::steady_clock::now();
            MV_FRAME_OUT frame = {0};
            const int ret = context->camera->GetImageBuffer(&frame, config.camera_settings.trigger_timeout_ms);
            const auto frame_ready = std::chrono::steady_clock::now();
            result.wait_frame_ms =
                std::chrono::duration<double, std::milli>(frame_ready - worker_start).count();

            if (ret == MV_OK) {
                const MV_FRAME_OUT_INFO_EX& frame_info = frame.stFrameInfo;
                result.host_timestamp = frame_ready;
                result.width = FrameWidth(frame_info);
                result.height = FrameHeight(frame_info);
                result.frame_index = frame_info.nFrameNum;
                result.device_timestamp =
                    (static_cast<std::uint64_t>(frame_info.nDevTimeStampHigh) << 32)
                    | frame_info.nDevTimeStampLow;
                result.sdk_host_timestamp = frame_info.nHostTimeStamp;
                result.pixel_format = PixelTypeToString(frame_info.enPixelType);
                result.frame_bytes = FrameLengthBytes(frame_info);

                try {
                    if (config.camera_settings.include_image_data) {
                        const auto image_process_start = std::chrono::steady_clock::now();
                        ConvertFrameToBgr(*context->camera, frame, result.bgr_image);
                        const auto image_process_end = std::chrono::steady_clock::now();
                        result.image_process_ms = std::chrono::duration<double, std::milli>(
                            image_process_end - image_process_start)
                            .count();
                    }
                    result.valid = true;
                } catch (const std::exception& ex) {
                    result.error_message = ex.what();
                }

                const int free_ret = context->camera->FreeImageBuffer(&frame);
                if (free_ret != MV_OK && result.error_message.empty()) {
                    result.error_message = FormatMvError(
                        "free frame buffer on camera[" + std::to_string(camera_index) + "]",
                        free_ret);
                    result.valid = false;
                }
            } else {
                result.error_message = FormatMvError(
                    "capture frame on camera[" + std::to_string(camera_index) + "]",
                    ret);
            }
            result.total_worker_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - worker_start)
                .count();

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
        last_capture_timing = {};
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

        const auto capture_start = std::chrono::steady_clock::now();
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

        const auto trigger_start = std::chrono::steady_clock::now();
        stereo_frame.trigger_timestamp = std::chrono::steady_clock::now();
        for (const auto& device : devices) {
            DeviceContext& context = *device;
            ThrowIfMvFailed(
                "software trigger on " + context.descriptor.serial_number,
                context.camera->CommandExecute("TriggerSoftware"));
        }
        const auto trigger_end = std::chrono::steady_clock::now();

        for (std::size_t camera_index = 0; camera_index < devices.size(); ++camera_index) {
            DeviceContext& context = *devices[camera_index];
            std::unique_lock<std::mutex> lock(context.mutex);
            context.completion_cv.wait(lock, [&]() {
                return context.completed_capture_seq >= capture_seq;
            });
        }

        const auto assemble_start = std::chrono::steady_clock::now();
        StereoCaptureTimingInfo timing_info;
        timing_info.capture_index = capture_seq;
        timing_info.trigger_ms =
            std::chrono::duration<double, std::milli>(trigger_end - trigger_start).count();
        for (std::size_t camera_index = 0; camera_index < devices.size(); ++camera_index) {
            DeviceContext& context = *devices[camera_index];
            std::lock_guard<std::mutex> lock(context.mutex);
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
            frame.bgr_image = context.last_result.bgr_image;
            stereo_frame.views[camera_index] = std::move(frame);

            timing_info.cameras[camera_index].pixel_format = context.last_result.pixel_format;
            timing_info.cameras[camera_index].frame_bytes = context.last_result.frame_bytes;
            timing_info.cameras[camera_index].wait_frame_ms = context.last_result.wait_frame_ms;
            timing_info.cameras[camera_index].image_process_ms = context.last_result.image_process_ms;
            timing_info.cameras[camera_index].total_worker_ms = context.last_result.total_worker_ms;
        }
        const auto capture_end = std::chrono::steady_clock::now();
        timing_info.assemble_ms =
            std::chrono::duration<double, std::milli>(capture_end - assemble_start).count();
        timing_info.total_capture_ms =
            std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
        last_capture_timing = timing_info;

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

std::array<StereoCaptureRuntimeInfo, 2> StereoCapture::RuntimeInfo() const {
    return impl_->runtime_info;
}

StereoCaptureTimingInfo StereoCapture::LastCaptureTiming() const {
    return impl_->last_capture_timing;
}

}  // namespace newnewhand
