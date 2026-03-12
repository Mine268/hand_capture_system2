#include "wilor_model.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace newnewhand {

WilorModel::WilorModel(const std::string& model_path, bool use_gpu, const std::string& profile_prefix)
    : env_(ORT_LOGGING_LEVEL_WARNING, "newnewhand_wilor"),
      session_(nullptr) {
    if (model_path.empty()) {
        throw std::invalid_argument("WiLoR model path must not be empty");
    }

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!profile_prefix.empty()) {
        options.EnableProfiling(profile_prefix.c_str());
        profiling_enabled_ = true;
    }

    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
    }

    session_ = Ort::Session(env_, model_path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = session_.GetInputCount();
    const std::size_t output_count = session_.GetOutputCount();

    input_names_.reserve(input_count);
    output_names_.reserve(output_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        auto name = session_.GetInputNameAllocated(index, allocator);
        input_names_.push_back(name.get());
    }
    for (std::size_t index = 0; index < output_count; ++index) {
        auto name = session_.GetOutputNameAllocated(index, allocator);
        output_names_.push_back(name.get());
    }
}

WilorModel::~WilorModel() {
    if (profiling_enabled_) {
        Ort::AllocatorWithDefaultOptions allocator;
        try {
            auto profile_path = session_.EndProfilingAllocated(allocator);
            if (profile_path) {
                std::cerr << "ORT profile saved: " << profile_path.get() << "\n";
            }
        } catch (...) {
        }
    }
}

WilorOutput WilorModel::Infer(const std::vector<cv::Mat>& patches) {
    const int batch_size = static_cast<int>(patches.size());
    if (batch_size == 0) {
        return {};
    }

    WilorOutput output;
    output.batch_size = batch_size;
    output.pred_cam.reserve(static_cast<std::size_t>(batch_size) * 3);
    output.global_orient.reserve(static_cast<std::size_t>(batch_size) * 3);
    output.global_orient_rotmat.reserve(static_cast<std::size_t>(batch_size) * 9);
    output.hand_pose.reserve(static_cast<std::size_t>(batch_size) * 45);
    output.hand_pose_rotmat.reserve(static_cast<std::size_t>(batch_size) * 135);
    output.betas.reserve(static_cast<std::size_t>(batch_size) * 10);
    output.pred_keypoints_3d.reserve(static_cast<std::size_t>(batch_size) * 63);
    output.pred_vertices.reserve(static_cast<std::size_t>(batch_size) * 2334);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    input_names.reserve(input_names_.size());
    output_names.reserve(output_names_.size());
    for (const auto& name : input_names_) {
        input_names.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
        output_names.push_back(name.c_str());
    }

    std::vector<float> input_data(256 * 256 * 3);
    const std::array<int64_t, 4> input_shape = {1, 256, 256, 3};

    for (const cv::Mat& patch : patches) {
        if (patch.empty() || patch.rows != 256 || patch.cols != 256 || patch.type() != CV_8UC3) {
            throw std::invalid_argument("each WiLoR patch must be a 256x256 BGR CV_8UC3 image");
        }

        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                const cv::Vec3b& pixel = patch.at<cv::Vec3b>(y, x);
                const int offset = (y * 256 + x) * 3;
                input_data[offset + 0] = static_cast<float>(pixel[0]);
                input_data[offset + 1] = static_cast<float>(pixel[1]);
                input_data[offset + 2] = static_cast<float>(pixel[2]);
            }
        }

        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            output_names.size());

        for (std::size_t output_index = 0; output_index < output_names_.size(); ++output_index) {
            const float* data = output_tensors[output_index].GetTensorData<float>();
            const std::size_t element_count =
                output_tensors[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
            const std::string& name = output_names_[output_index];

            std::vector<float>* target = nullptr;
            if (name == "pred_cam") {
                target = &output.pred_cam;
            } else if (name == "global_orient") {
                target = &output.global_orient;
            } else if (name == "global_orient_rotmat") {
                target = &output.global_orient_rotmat;
            } else if (name == "hand_pose") {
                target = &output.hand_pose;
            } else if (name == "hand_pose_rotmat") {
                target = &output.hand_pose_rotmat;
            } else if (name == "betas") {
                target = &output.betas;
            } else if (name == "pred_keypoints_3d") {
                target = &output.pred_keypoints_3d;
            } else if (name == "pred_vertices") {
                target = &output.pred_vertices;
            }

            if (target) {
                target->insert(target->end(), data, data + element_count);
            }
        }
    }

    return output;
}

}  // namespace newnewhand
