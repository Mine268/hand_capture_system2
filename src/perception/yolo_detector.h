#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

namespace newnewhand {

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int class_id = 0;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, bool use_gpu);

    std::vector<Detection> Detect(
        const cv::Mat& image,
        float confidence_threshold,
        float nms_threshold);

private:
    cv::Mat Letterbox(const cv::Mat& src, int target_size, float& scale, int& pad_x, int& pad_y) const;
    std::vector<Detection> Postprocess(
        const float* output,
        int num_candidates,
        float confidence_threshold,
        float nms_threshold,
        float scale,
        int pad_x,
        int pad_y) const;
    void ApplyNms(std::vector<Detection>& detections, float nms_threshold) const;

    Ort::Env env_;
    Ort::Session session_;
    int input_size_ = 512;
};

}  // namespace newnewhand
