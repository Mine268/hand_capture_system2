#include "yolo_detector.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace newnewhand {

YoloDetector::YoloDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "newnewhand_yolo"),
      session_(nullptr) {
    if (model_path.empty()) {
        throw std::invalid_argument("detector model path must not be empty");
    }

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), options);
}

cv::Mat YoloDetector::Letterbox(const cv::Mat& src, int target_size, float& scale, int& pad_x, int& pad_y) const {
    const int height = src.rows;
    const int width = src.cols;
    scale = std::min(
        static_cast<float>(target_size) / static_cast<float>(height),
        static_cast<float>(target_size) / static_cast<float>(width));

    const int resized_width = static_cast<int>(width * scale);
    const int resized_height = static_cast<int>(height * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_width, resized_height));

    pad_x = (target_size - resized_width) / 2;
    pad_y = (target_size - resized_height) / 2;

    cv::Mat padded(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    return padded;
}

std::vector<Detection> YoloDetector::Detect(
    const cv::Mat& image,
    float confidence_threshold,
    float nms_threshold) {
    if (image.empty()) {
        throw std::invalid_argument("input image must not be empty");
    }

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    const cv::Mat padded = Letterbox(image, input_size_, scale, pad_x, pad_y);

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    cv::Mat blob;
    rgb.convertTo(blob, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_data(3 * input_size_ * input_size_);
    const float* src_ptr = reinterpret_cast<const float*>(blob.data);
    for (int channel = 0; channel < 3; ++channel) {
        for (int pixel_index = 0; pixel_index < input_size_ * input_size_; ++pixel_index) {
            input_data[channel * input_size_ * input_size_ + pixel_index] =
                src_ptr[pixel_index * 3 + channel];
        }
    }

    const std::array<int64_t, 4> input_shape = {1, 3, input_size_, input_size_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    const auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_candidates = static_cast<int>(shape.at(2));
    const float* output_data = output_tensors[0].GetTensorData<float>();

    return Postprocess(
        output_data,
        num_candidates,
        confidence_threshold,
        nms_threshold,
        scale,
        pad_x,
        pad_y);
}

std::vector<Detection> YoloDetector::Postprocess(
    const float* output,
    int num_candidates,
    float confidence_threshold,
    float nms_threshold,
    float scale,
    int pad_x,
    int pad_y) const {
    std::vector<Detection> detections;
    detections.reserve(static_cast<std::size_t>(num_candidates));

    for (int candidate_index = 0; candidate_index < num_candidates; ++candidate_index) {
        const float cx = output[0 * num_candidates + candidate_index];
        const float cy = output[1 * num_candidates + candidate_index];
        const float width = output[2 * num_candidates + candidate_index];
        const float height = output[3 * num_candidates + candidate_index];
        const float left_score = output[4 * num_candidates + candidate_index];
        const float right_score = output[5 * num_candidates + candidate_index];

        const float confidence = std::max(left_score, right_score);
        if (confidence < confidence_threshold) {
            continue;
        }

        const int class_id = right_score > left_score ? 1 : 0;
        detections.push_back({
            (cx - width * 0.5f - pad_x) / scale,
            (cy - height * 0.5f - pad_y) / scale,
            (cx + width * 0.5f - pad_x) / scale,
            (cy + height * 0.5f - pad_y) / scale,
            confidence,
            class_id});
    }

    ApplyNms(detections, nms_threshold);
    return detections;
}

void YoloDetector::ApplyNms(std::vector<Detection>& detections, float nms_threshold) const {
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& lhs, const Detection& rhs) {
            return lhs.confidence > rhs.confidence;
        });

    std::vector<Detection> filtered;
    std::vector<bool> suppressed(detections.size(), false);
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        filtered.push_back(detections[i]);
        for (std::size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            const float ix1 = std::max(detections[i].x1, detections[j].x1);
            const float iy1 = std::max(detections[i].y1, detections[j].y1);
            const float ix2 = std::min(detections[i].x2, detections[j].x2);
            const float iy2 = std::min(detections[i].y2, detections[j].y2);
            const float intersection_w = std::max(0.0f, ix2 - ix1);
            const float intersection_h = std::max(0.0f, iy2 - iy1);
            const float intersection = intersection_w * intersection_h;
            const float area_i =
                (detections[i].x2 - detections[i].x1) * (detections[i].y2 - detections[i].y1);
            const float area_j =
                (detections[j].x2 - detections[j].x1) * (detections[j].y2 - detections[j].y1);
            const float iou = intersection / (area_i + area_j - intersection + 1e-9f);
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(filtered);
}

}  // namespace newnewhand
