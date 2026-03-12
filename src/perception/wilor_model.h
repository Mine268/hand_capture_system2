#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

namespace newnewhand {

struct WilorOutput {
    int batch_size = 0;
    std::vector<float> pred_cam;
    std::vector<float> global_orient;
    std::vector<float> global_orient_rotmat;
    std::vector<float> hand_pose;
    std::vector<float> hand_pose_rotmat;
    std::vector<float> betas;
    std::vector<float> pred_keypoints_3d;
    std::vector<float> pred_vertices;
};

class WilorModel {
public:
    WilorModel(
        const std::string& network_model_path,
        const std::string& mano_model_path,
        bool use_gpu,
        const std::string& profile_prefix = {});
    ~WilorModel();

    WilorOutput Infer(const std::vector<cv::Mat>& patches);
    void FillCpuManoGeometry(WilorOutput& output);

private:
    void AppendOutputBuffer(
        WilorOutput& output,
        const std::string& name,
        const float* data,
        std::size_t element_count);

    void RunEmbeddedManoNetwork(
        WilorOutput& output,
        const std::vector<const char*>& output_names,
        const std::vector<Ort::Value>& output_tensors);

    void RunCpuMano(
        WilorOutput& output,
        const float* global_orient_rotmat,
        const float* hand_pose_rotmat,
        const float* betas);

    Ort::Env env_;
    Ort::Session network_session_;
    std::unique_ptr<Ort::Session> mano_session_;
    std::vector<std::string> network_input_names_;
    std::vector<std::string> network_output_names_;
    std::vector<std::string> mano_input_names_;
    std::vector<std::string> mano_output_names_;
    bool profiling_enabled_ = false;
    bool network_outputs_vertices_ = false;
};

}  // namespace newnewhand
