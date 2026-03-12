#include "wilor_model.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace newnewhand {

WilorModel::WilorModel(
    const std::string& network_model_path,
    const std::string& mano_model_path,
    bool use_gpu,
    const std::string& profile_prefix)
    : env_(ORT_LOGGING_LEVEL_WARNING, "newnewhand_wilor"),
      network_session_(nullptr) {
    if (network_model_path.empty()) {
        throw std::invalid_argument("WiLoR network model path must not be empty");
    }

    Ort::SessionOptions network_options;
    network_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!profile_prefix.empty()) {
        network_options.EnableProfiling(profile_prefix.c_str());
        profiling_enabled_ = true;
    }

    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        network_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    network_session_ = Ort::Session(env_, network_model_path.c_str(), network_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = network_session_.GetInputCount();
    const std::size_t output_count = network_session_.GetOutputCount();

    network_input_names_.reserve(input_count);
    network_output_names_.reserve(output_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        auto name = network_session_.GetInputNameAllocated(index, allocator);
        network_input_names_.push_back(name.get());
    }
    for (std::size_t index = 0; index < output_count; ++index) {
        auto name = network_session_.GetOutputNameAllocated(index, allocator);
        network_output_names_.push_back(name.get());
        if (network_output_names_.back() == "pred_keypoints_3d" || network_output_names_.back() == "pred_vertices") {
            network_outputs_vertices_ = true;
        }
    }

    if (!network_outputs_vertices_) {
        if (mano_model_path.empty()) {
            throw std::invalid_argument("MANO CPU model path must not be empty when WiLoR network outputs only MANO params");
        }

        Ort::SessionOptions mano_options;
        mano_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        mano_session_ = std::make_unique<Ort::Session>(env_, mano_model_path.c_str(), mano_options);

        const std::size_t mano_input_count = mano_session_->GetInputCount();
        const std::size_t mano_output_count = mano_session_->GetOutputCount();
        mano_input_names_.reserve(mano_input_count);
        mano_output_names_.reserve(mano_output_count);
        for (std::size_t index = 0; index < mano_input_count; ++index) {
            auto name = mano_session_->GetInputNameAllocated(index, allocator);
            mano_input_names_.push_back(name.get());
        }
        for (std::size_t index = 0; index < mano_output_count; ++index) {
            auto name = mano_session_->GetOutputNameAllocated(index, allocator);
            mano_output_names_.push_back(name.get());
        }
    }
}

WilorModel::~WilorModel() {
    if (profiling_enabled_) {
        Ort::AllocatorWithDefaultOptions allocator;
        try {
            auto profile_path = network_session_.EndProfilingAllocated(allocator);
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
    std::vector<const char*> network_input_names;
    std::vector<const char*> network_output_names;
    network_input_names.reserve(network_input_names_.size());
    network_output_names.reserve(network_output_names_.size());
    for (const auto& name : network_input_names_) {
        network_input_names.push_back(name.c_str());
    }
    for (const auto& name : network_output_names_) {
        network_output_names.push_back(name.c_str());
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

        auto output_tensors = network_session_.Run(
            Ort::RunOptions{nullptr},
            network_input_names.data(),
            &input_tensor,
            1,
            network_output_names.data(),
            network_output_names.size());

        if (network_outputs_vertices_) {
            RunEmbeddedManoNetwork(output, network_output_names, output_tensors);
        } else {
            const float* global_orient_rotmat = nullptr;
            const float* hand_pose_rotmat = nullptr;
            const float* betas = nullptr;

            for (std::size_t output_index = 0; output_index < network_output_names_.size(); ++output_index) {
                const float* data = output_tensors[output_index].GetTensorData<float>();
                const std::size_t element_count =
                    output_tensors[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
                const std::string& name = network_output_names_[output_index];
                AppendOutputBuffer(output, name, data, element_count);

                if (name == "global_orient_rotmat") {
                    global_orient_rotmat = data;
                } else if (name == "hand_pose_rotmat") {
                    hand_pose_rotmat = data;
                } else if (name == "betas") {
                    betas = data;
                }
            }

            if (!global_orient_rotmat || !hand_pose_rotmat || !betas) {
                throw std::runtime_error("split WiLoR model did not return required MANO parameter tensors");
            }
            RunCpuMano(output, global_orient_rotmat, hand_pose_rotmat, betas);
        }
    }

    return output;
}

void WilorModel::AppendOutputBuffer(
    WilorOutput& output,
    const std::string& name,
    const float* data,
    std::size_t element_count) {
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

void WilorModel::RunEmbeddedManoNetwork(
    WilorOutput& output,
    const std::vector<const char*>& output_names,
    const std::vector<Ort::Value>& output_tensors) {
    for (std::size_t output_index = 0; output_index < output_names.size(); ++output_index) {
        const float* data = output_tensors[output_index].GetTensorData<float>();
        const std::size_t element_count =
            output_tensors[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
        AppendOutputBuffer(output, network_output_names_[output_index], data, element_count);
    }
}

void WilorModel::RunCpuMano(
    WilorOutput& output,
    const float* global_orient_rotmat,
    const float* hand_pose_rotmat,
    const float* betas) {
    if (!mano_session_) {
        throw std::runtime_error("MANO CPU session is not initialized");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> global_shape = {1, 1, 3, 3};
    std::array<int64_t, 4> hand_shape = {1, 15, 3, 3};
    std::array<int64_t, 2> betas_shape = {1, 10};

    auto global_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(global_orient_rotmat),
        9,
        global_shape.data(),
        global_shape.size());
    auto hand_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(hand_pose_rotmat),
        15 * 9,
        hand_shape.data(),
        hand_shape.size());
    auto betas_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(betas),
        10,
        betas_shape.data(),
        betas_shape.size());

    std::array<Ort::Value, 3> input_tensors = {
        std::move(global_tensor),
        std::move(hand_tensor),
        std::move(betas_tensor),
    };

    std::vector<const char*> mano_input_names;
    std::vector<const char*> mano_output_names;
    mano_input_names.reserve(mano_input_names_.size());
    mano_output_names.reserve(mano_output_names_.size());
    for (const auto& name : mano_input_names_) {
        mano_input_names.push_back(name.c_str());
    }
    for (const auto& name : mano_output_names_) {
        mano_output_names.push_back(name.c_str());
    }

    auto output_tensors = mano_session_->Run(
        Ort::RunOptions{nullptr},
        mano_input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        mano_output_names.data(),
        mano_output_names.size());

    for (std::size_t output_index = 0; output_index < mano_output_names_.size(); ++output_index) {
        const float* data = output_tensors[output_index].GetTensorData<float>();
        const std::size_t element_count =
            output_tensors[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
        AppendOutputBuffer(output, mano_output_names_[output_index], data, element_count);
    }
}

}  // namespace newnewhand
