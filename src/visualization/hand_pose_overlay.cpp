#include "newnewhand/visualization/hand_pose_overlay.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace newnewhand {

namespace {

struct ManoFace {
    int v0 = 0;
    int v1 = 0;
    int v2 = 0;
};

struct RenderTriangle {
    std::array<cv::Point, 3> points;
    cv::Scalar color;
    float depth = 0.0f;
};

constexpr int kHandConnections[20][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    {0, 17}, {17, 18}, {18, 19}, {19, 20},
};

const cv::Scalar kFingerColors[5] = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
};

cv::Scalar HandMeshBaseColor(bool is_right) {
    return is_right ? cv::Scalar(255, 190, 40) : cv::Scalar(220, 90, 255);
}

std::string DefaultManoFacePath() {
#ifdef NEWNEWHAND_PROJECT_ROOT
    return std::string(NEWNEWHAND_PROJECT_ROOT) + "/resources/mano_faces.txt";
#else
    return {};
#endif
}

const std::vector<ManoFace>& LoadManoFaces() {
    static std::once_flag once_flag;
    static std::vector<ManoFace> faces;

    std::call_once(once_flag, []() {
        const std::string path = DefaultManoFacePath();
        if (path.empty()) {
            return;
        }

        std::ifstream input(path);
        if (!input.is_open()) {
            return;
        }

        ManoFace face;
        while (input >> face.v0 >> face.v1 >> face.v2) {
            faces.push_back(face);
        }
    });

    return faces;
}

cv::Scalar ShadeColor(const cv::Scalar& base_color, float brightness) {
    const double scale = std::clamp(static_cast<double>(brightness), 0.25, 1.0);
    return cv::Scalar(base_color[0] * scale, base_color[1] * scale, base_color[2] * scale);
}

cv::Vec3f RotatePoint(const cv::Vec3f& point, float yaw_degrees, float pitch_degrees, float roll_degrees) {
    const float yaw = yaw_degrees * static_cast<float>(CV_PI) / 180.0f;
    const float pitch = pitch_degrees * static_cast<float>(CV_PI) / 180.0f;
    const float roll = roll_degrees * static_cast<float>(CV_PI) / 180.0f;

    const float cosy = std::cos(yaw);
    const float siny = std::sin(yaw);
    const float cosp = std::cos(pitch);
    const float sinp = std::sin(pitch);
    const float cosr = std::cos(roll);
    const float sinr = std::sin(roll);

    cv::Vec3f x = point;
    x = cv::Vec3f(cosy * x[0] + siny * x[2], x[1], -siny * x[0] + cosy * x[2]);
    x = cv::Vec3f(x[0], cosp * x[1] - sinp * x[2], sinp * x[1] + cosp * x[2]);
    x = cv::Vec3f(cosr * x[0] - sinr * x[1], sinr * x[0] + cosr * x[1], x[2]);
    return x;
}

void RenderHandMesh(
    cv::Mat& bgr_image,
    const HandPoseResult& result,
    const HandPoseOverlayStyle& style) {
    const auto& faces = LoadManoFaces();
    if (!style.draw_mesh || faces.empty() || result.focal_length_px <= 0.0f) {
        return;
    }

    constexpr int kNumVertices = 778;
    std::array<cv::Point2f, kNumVertices> projected_vertices;
    std::array<cv::Vec3f, kNumVertices> camera_vertices;
    const float image_cx = static_cast<float>(bgr_image.cols) * 0.5f;
    const float image_cy = static_cast<float>(bgr_image.rows) * 0.5f;

    for (int vertex_index = 0; vertex_index < kNumVertices; ++vertex_index) {
        const float x = result.vertices[vertex_index][0] + result.camera_translation[0];
        const float y = result.vertices[vertex_index][1] + result.camera_translation[1];
        const float z = result.vertices[vertex_index][2] + result.camera_translation[2];
        camera_vertices[vertex_index] = cv::Vec3f(x, y, z);
        if (z <= 1e-6f) {
            projected_vertices[vertex_index] = cv::Point2f(-1.0f, -1.0f);
            continue;
        }

        projected_vertices[vertex_index] = cv::Point2f(
            result.focal_length_px * (x / z) + image_cx,
            result.focal_length_px * (y / z) + image_cy);
    }

    std::vector<RenderTriangle> triangles;
    triangles.reserve(faces.size());
    const cv::Scalar base_color = HandMeshBaseColor(result.detection.is_right);

    for (const auto& face : faces) {
        const cv::Vec3f v0 = camera_vertices[face.v0];
        const cv::Vec3f v1 = camera_vertices[face.v1];
        const cv::Vec3f v2 = camera_vertices[face.v2];
        if (v0[2] <= 1e-6f || v1[2] <= 1e-6f || v2[2] <= 1e-6f) {
            continue;
        }

        const cv::Point2f p0 = projected_vertices[face.v0];
        const cv::Point2f p1 = projected_vertices[face.v1];
        const cv::Point2f p2 = projected_vertices[face.v2];
        if (p0.x < 0.0f || p1.x < 0.0f || p2.x < 0.0f) {
            continue;
        }

        const cv::Vec3f e1 = v1 - v0;
        const cv::Vec3f e2 = v2 - v0;
        const cv::Vec3f normal = e1.cross(e2);
        const float normal_norm = cv::norm(normal);
        const float facing = normal_norm > 1e-6f ? std::abs(normal[2]) / normal_norm : 0.5f;
        const float brightness = 0.25f + 0.75f * facing;

        RenderTriangle triangle;
        triangle.points = {
            cv::Point(cvRound(p0.x), cvRound(p0.y)),
            cv::Point(cvRound(p1.x), cvRound(p1.y)),
            cv::Point(cvRound(p2.x), cvRound(p2.y)),
        };
        triangle.color = ShadeColor(base_color, brightness);
        triangle.depth = (v0[2] + v1[2] + v2[2]) / 3.0f;
        triangles.push_back(triangle);
    }

    std::sort(
        triangles.begin(),
        triangles.end(),
        [](const RenderTriangle& lhs, const RenderTriangle& rhs) {
            return lhs.depth > rhs.depth;
        });

    cv::Mat mesh_layer = bgr_image.clone();
    cv::Mat mask = cv::Mat::zeros(bgr_image.size(), CV_8UC1);
    for (const auto& triangle : triangles) {
        const cv::Point polygon[3] = {triangle.points[0], triangle.points[1], triangle.points[2]};
        cv::fillConvexPoly(mesh_layer, polygon, 3, triangle.color, cv::LINE_AA);
        cv::fillConvexPoly(mask, polygon, 3, cv::Scalar(255), cv::LINE_AA);
        if (style.draw_mesh_wireframe) {
            const cv::Point* polygons[] = {polygon};
            const int polygon_size[] = {3};
            cv::polylines(
                mesh_layer,
                polygons,
                polygon_size,
                1,
                true,
                cv::Scalar(20, 20, 20),
                1,
                cv::LINE_AA);
        }
    }

    cv::Mat blended;
    cv::addWeighted(mesh_layer, style.mesh_alpha, bgr_image, 1.0 - style.mesh_alpha, 0.0, blended);
    blended.copyTo(bgr_image, mask);
}

}  // namespace

void DrawHandPoseOverlay(
    cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style) {
    for (const auto& result : results) {
        RenderHandMesh(bgr_image, result, style);

        if (style.draw_bbox) {
            cv::rectangle(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.detection.bbox[0]),
                    static_cast<int>(result.detection.bbox[1])),
                cv::Point(
                    static_cast<int>(result.detection.bbox[2]),
                    static_cast<int>(result.detection.bbox[3])),
                cv::Scalar(0, 255, 0),
                style.line_thickness);
        }

        if (style.draw_label) {
            const std::string label = result.detection.is_right ? "R" : "L";
            cv::putText(
                bgr_image,
                label,
                cv::Point(
                    static_cast<int>(result.detection.bbox[0]),
                    static_cast<int>(result.detection.bbox[1]) - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                style.label_scale,
                cv::Scalar(0, 255, 0),
                style.line_thickness);
        }

        for (int connection_index = 0; connection_index < 20; ++connection_index) {
            const int start_joint = kHandConnections[connection_index][0];
            const int end_joint = kHandConnections[connection_index][1];
            cv::line(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[start_joint][0]),
                    static_cast<int>(result.keypoints_2d[start_joint][1])),
                cv::Point(
                    static_cast<int>(result.keypoints_2d[end_joint][0]),
                    static_cast<int>(result.keypoints_2d[end_joint][1])),
                kFingerColors[connection_index / 4],
                style.line_thickness);
        }

        for (int joint_index = 0; joint_index < 21; ++joint_index) {
            cv::circle(
                bgr_image,
                cv::Point(
                    static_cast<int>(result.keypoints_2d[joint_index][0]),
                    static_cast<int>(result.keypoints_2d[joint_index][1])),
                style.joint_radius,
                cv::Scalar(0, 0, 255),
                -1);
        }
    }
}

cv::Mat RenderHandPoseOverlay(
    const cv::Mat& bgr_image,
    const std::vector<HandPoseResult>& results,
    const HandPoseOverlayStyle& style) {
    cv::Mat rendered = bgr_image.clone();
    DrawHandPoseOverlay(rendered, results, style);
    return rendered;
}

}  // namespace newnewhand
