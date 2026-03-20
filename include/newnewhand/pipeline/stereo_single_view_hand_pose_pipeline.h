#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "newnewhand/capture/stereo_capture.h"
#include "newnewhand/perception/hand_pose_estimator.h"

namespace newnewhand {

struct SingleViewPoseView {
    CameraFrame camera_frame;
    std::vector<HandPoseResult> hand_poses;
    cv::Mat overlay_image;
    std::string inference_error;
    bool used_cpu_fallback = false;
};

struct StereoSingleViewPoseFrame {
    std::uint64_t capture_index = 0;
    std::chrono::steady_clock::time_point trigger_timestamp;
    std::array<SingleViewPoseView, 2> views;

    bool is_complete() const {
        return views[0].camera_frame.valid && views[1].camera_frame.valid;
    }
};

struct StereoSingleViewHandPosePipelineConfig {
    StereoCaptureConfig capture_config;
    HandPoseEstimatorConfig pose_config;
    bool continue_on_inference_error = true;
    bool retry_with_cpu_on_error = true;
};

class StereoSingleViewHandPosePipeline {
public:
    explicit StereoSingleViewHandPosePipeline(StereoSingleViewHandPosePipelineConfig config);
    ~StereoSingleViewHandPosePipeline();

    StereoSingleViewHandPosePipeline(const StereoSingleViewHandPosePipeline&) = delete;
    StereoSingleViewHandPosePipeline& operator=(const StereoSingleViewHandPosePipeline&) = delete;

    StereoSingleViewHandPosePipeline(StereoSingleViewHandPosePipeline&&) noexcept;
    StereoSingleViewHandPosePipeline& operator=(StereoSingleViewHandPosePipeline&&) noexcept;

    void Initialize();
    void Start();
    StereoFrame Capture();
    StereoSingleViewPoseFrame CaptureAndEstimate();
    StereoSingleViewPoseFrame Estimate(const StereoFrame& stereo_frame);
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
