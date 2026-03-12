#pragma once

#include <array>

#include <opencv2/core/mat.hpp>

namespace newnewhand::hand_pose_utils {

cv::Mat GenerateImagePatch(
    const cv::Mat& image,
    float cx,
    float cy,
    float bbox_size,
    int patch_size,
    bool do_flip);

std::array<float, 3> CamCropToFull(
    const float* cam_bbox,
    float cx,
    float cy,
    float box_size,
    float image_width,
    float image_height,
    float focal_length);

void PerspectiveProjection(
    const float* points_3d,
    int num_points,
    const float* translation,
    float focal_length,
    float cx,
    float cy,
    float* points_2d);

void RotationMatrixToRotationVector(const float* rotation_matrix, float* rotation_vector);
cv::Vec3f RotationMatrixToRotationVector(const cv::Matx33f& rotation_matrix);
cv::Matx33f RotationVectorToRotationMatrix(const cv::Vec3f& rotation_vector);

cv::Mat AntiAliasBlur(const cv::Mat& image, float bbox_size, int patch_size);

}  // namespace newnewhand::hand_pose_utils
