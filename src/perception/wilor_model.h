#pragma once

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
    WilorModel(const std::string& model_path, bool use_gpu, const std::string& profile_prefix = {});
    ~WilorModel();

    WilorOutput Infer(const std::vector<cv::Mat>& patches);

private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    bool profiling_enabled_ = false;
};

}  // namespace newnewhand
