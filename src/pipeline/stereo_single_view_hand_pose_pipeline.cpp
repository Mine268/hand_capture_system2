#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"

#include <memory>
#include <string>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "newnewhand/visualization/hand_pose_overlay.h"

namespace newnewhand {

namespace {

void DrawInferenceError(cv::Mat& image, const std::string& error_message) {
    if (image.empty() || error_message.empty()) {
        return;
    }

    cv::putText(
        image,
        "Inference error",
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(0, 0, 255),
        2);
    cv::putText(
        image,
        error_message.substr(0, 120),
        cv::Point(20, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(0, 0, 255),
        1);
}

}  // namespace

struct StereoSingleViewHandPosePipeline::Impl {
    explicit Impl(StereoSingleViewHandPosePipelineConfig config_in)
        : config(std::move(config_in)),
          capture(config.capture_config),
          pose_estimator(config.pose_config) {
        if (config.pose_config.use_gpu && config.retry_with_cpu_on_error) {
            HandPoseEstimatorConfig fallback_config = config.pose_config;
            fallback_config.use_gpu = false;
            cpu_fallback_estimator = std::make_unique<HandPoseEstimator>(std::move(fallback_config));
        }
    }

    StereoSingleViewHandPosePipelineConfig config;
    StereoCapture capture;
    HandPoseEstimator pose_estimator;
    std::unique_ptr<HandPoseEstimator> cpu_fallback_estimator;
};

StereoSingleViewHandPosePipeline::StereoSingleViewHandPosePipeline(
    StereoSingleViewHandPosePipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

StereoSingleViewHandPosePipeline::~StereoSingleViewHandPosePipeline() {
    if (impl_) {
        impl_->capture.Shutdown();
    }
}

StereoSingleViewHandPosePipeline::StereoSingleViewHandPosePipeline(
    StereoSingleViewHandPosePipeline&&) noexcept = default;

StereoSingleViewHandPosePipeline& StereoSingleViewHandPosePipeline::operator=(
    StereoSingleViewHandPosePipeline&&) noexcept = default;

void StereoSingleViewHandPosePipeline::Initialize() {
    impl_->capture.Initialize();
}

void StereoSingleViewHandPosePipeline::Start() {
    impl_->capture.Start();
}

StereoSingleViewPoseFrame StereoSingleViewHandPosePipeline::CaptureAndEstimate() {
    return Estimate(impl_->capture.Capture());
}

StereoSingleViewPoseFrame StereoSingleViewHandPosePipeline::Estimate(const StereoFrame& stereo_frame) {
    StereoSingleViewPoseFrame pose_frame;
    pose_frame.capture_index = stereo_frame.capture_index;
    pose_frame.trigger_timestamp = stereo_frame.trigger_timestamp;

    for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
        SingleViewPoseView view;
        view.camera_frame = stereo_frame.views[camera_index];
        if (view.camera_frame.valid && !view.camera_frame.bgr_image.empty()) {
            view.overlay_image = view.camera_frame.bgr_image.clone();
            try {
                view.hand_poses = impl_->pose_estimator.Predict(view.camera_frame.bgr_image);
            } catch (const std::exception& primary_error) {
                if (impl_->cpu_fallback_estimator) {
                    try {
                        view.hand_poses = impl_->cpu_fallback_estimator->Predict(view.camera_frame.bgr_image);
                        view.used_cpu_fallback = true;
                    } catch (const std::exception& fallback_error) {
                        view.inference_error =
                            std::string("primary: ") + primary_error.what()
                            + " | cpu fallback: " + fallback_error.what();
                    }
                } else {
                    view.inference_error = primary_error.what();
                }

                if (!view.inference_error.empty() && !impl_->config.continue_on_inference_error) {
                    throw;
                }
            }

            if (!view.hand_poses.empty()) {
                DrawHandPoseOverlay(view.overlay_image, view.hand_poses);
            }
            if (!view.inference_error.empty()) {
                DrawInferenceError(view.overlay_image, view.inference_error);
            }
        }
        pose_frame.views[camera_index] = std::move(view);
    }

    return pose_frame;
}

void StereoSingleViewHandPosePipeline::Stop() {
    impl_->capture.Stop();
}

void StereoSingleViewHandPosePipeline::Shutdown() {
    impl_->capture.Shutdown();
}

bool StereoSingleViewHandPosePipeline::IsInitialized() const {
    return impl_->capture.IsInitialized();
}

bool StereoSingleViewHandPosePipeline::IsRunning() const {
    return impl_->capture.IsRunning();
}

std::array<CameraDescriptor, 2> StereoSingleViewHandPosePipeline::ActiveCameras() const {
    return impl_->capture.ActiveCameras();
}

}  // namespace newnewhand
