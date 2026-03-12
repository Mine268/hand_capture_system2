#include "hand_pose_utils.h"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace newnewhand::hand_pose_utils {

cv::Mat GenerateImagePatch(
    const cv::Mat& image,
    float cx,
    float cy,
    float bbox_size,
    int patch_size,
    bool do_flip) {
    cv::Mat source_image;
    float center_x = cx;
    float center_y = cy;

    if (do_flip) {
        cv::flip(image, source_image, 1);
        center_x = static_cast<float>(image.cols) - cx - 1.0f;
    } else {
        source_image = image;
    }

    const float half_source = bbox_size * 0.5f;
    const float half_destination = static_cast<float>(patch_size) * 0.5f;

    cv::Point2f source_points[3] = {
        {center_x, center_y},
        {center_x, center_y + half_source},
        {center_x + half_source, center_y},
    };
    cv::Point2f destination_points[3] = {
        {half_destination, half_destination},
        {half_destination, static_cast<float>(patch_size)},
        {static_cast<float>(patch_size), half_destination},
    };

    const cv::Mat transform = cv::getAffineTransform(source_points, destination_points);
    cv::Mat patch;
    cv::warpAffine(
        source_image,
        patch,
        transform,
        cv::Size(patch_size, patch_size),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
    return patch;
}

std::array<float, 3> CamCropToFull(
    const float* cam_bbox,
    float cx,
    float cy,
    float box_size,
    float image_width,
    float image_height,
    float focal_length) {
    const float width_half = image_width * 0.5f;
    const float height_half = image_height * 0.5f;
    const float scaled_box = box_size * cam_bbox[0] + 1e-9f;
    const float tz = 2.0f * focal_length / scaled_box;
    const float tx = (2.0f * (cx - width_half) / scaled_box) + cam_bbox[1];
    const float ty = (2.0f * (cy - height_half) / scaled_box) + cam_bbox[2];
    return {tx, ty, tz};
}

void PerspectiveProjection(
    const float* points_3d,
    int num_points,
    const float* translation,
    float focal_length,
    float cx,
    float cy,
    float* points_2d) {
    for (int point_index = 0; point_index < num_points; ++point_index) {
        const float x = points_3d[point_index * 3 + 0] + translation[0];
        const float y = points_3d[point_index * 3 + 1] + translation[1];
        const float z = points_3d[point_index * 3 + 2] + translation[2];

        points_2d[point_index * 2 + 0] = focal_length * (x / z) + cx;
        points_2d[point_index * 2 + 1] = focal_length * (y / z) + cy;
    }
}

void RotationMatrixToRotationVector(const float* rotation_matrix, float* rotation_vector) {
    const float r00 = rotation_matrix[0];
    const float r01 = rotation_matrix[1];
    const float r02 = rotation_matrix[2];
    const float r10 = rotation_matrix[3];
    const float r11 = rotation_matrix[4];
    const float r12 = rotation_matrix[5];
    const float r20 = rotation_matrix[6];
    const float r21 = rotation_matrix[7];
    const float r22 = rotation_matrix[8];

    const float trace = r00 + r11 + r22;
    const float cos_theta = std::clamp((trace - 1.0f) * 0.5f, -1.0f, 1.0f);
    const float theta = std::acos(cos_theta);
    constexpr float kSmallAngle = 1e-5f;
    constexpr float kPiAngle = 1e-4f;

    if (theta < kSmallAngle) {
        rotation_vector[0] = 0.5f * (r21 - r12);
        rotation_vector[1] = 0.5f * (r02 - r20);
        rotation_vector[2] = 0.5f * (r10 - r01);
        return;
    }

    if (static_cast<float>(M_PI) - theta < kPiAngle) {
        float axis_x = std::sqrt(std::max(0.0f, (r00 + 1.0f) * 0.5f));
        float axis_y = std::sqrt(std::max(0.0f, (r11 + 1.0f) * 0.5f));
        float axis_z = std::sqrt(std::max(0.0f, (r22 + 1.0f) * 0.5f));

        if (r21 - r12 < 0.0f) {
            axis_x = -axis_x;
        }
        if (r02 - r20 < 0.0f) {
            axis_y = -axis_y;
        }
        if (r10 - r01 < 0.0f) {
            axis_z = -axis_z;
        }

        const float axis_norm = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
        if (axis_norm > kSmallAngle) {
            axis_x /= axis_norm;
            axis_y /= axis_norm;
            axis_z /= axis_norm;
        }

        rotation_vector[0] = theta * axis_x;
        rotation_vector[1] = theta * axis_y;
        rotation_vector[2] = theta * axis_z;
        return;
    }

    const float sin_theta = std::sin(theta);
    const float scale = theta / (2.0f * sin_theta);
    rotation_vector[0] = scale * (r21 - r12);
    rotation_vector[1] = scale * (r02 - r20);
    rotation_vector[2] = scale * (r10 - r01);
}

cv::Mat AntiAliasBlur(const cv::Mat& image, float bbox_size, int patch_size) {
    const float downsample_factor = (bbox_size / static_cast<float>(patch_size)) * 0.5f;
    if (downsample_factor <= 1.1f) {
        return image;
    }

    const float sigma = (downsample_factor - 1.0f) * 0.5f;
    int kernel_size = static_cast<int>(std::ceil(sigma * 6.0f)) | 1;
    if (kernel_size < 3) {
        kernel_size = 3;
    }

    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(kernel_size, kernel_size), sigma, sigma);
    return blurred;
}

}  // namespace newnewhand::hand_pose_utils
