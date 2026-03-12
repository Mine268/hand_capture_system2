#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include "newnewhand/fusion/stereo_hand_fuser.h"

struct GLFWwindow;

namespace newnewhand {

struct GlfwSceneViewerConfig {
    int width = 1280;
    int height = 900;
    std::string title = "newnewhand OpenGL Viewer";
    float camera_distance = 0.6f;
    float world_offset_x = 0.0f;
    float world_offset_y = 0.0f;
    float world_offset_z = 0.0f;
    float yaw_degrees = -35.0f;
    float pitch_degrees = 18.0f;
    float roll_degrees = 0.0f;
    float angular_speed_degrees = 70.0f;
    float translation_speed = 0.20f;
    float zoom_speed = 0.6f;
    float axis_length = 0.10f;
    bool draw_world_axes = false;
    bool draw_cam0_axes = true;
    bool draw_cam0_frustum = true;
    bool draw_cam1_axes = true;
    bool draw_cam1_frustum = true;
    bool draw_ground_grid = true;
    bool draw_mesh = true;
    bool draw_wireframe = false;

    bool has_cam1_pose = false;
    cv::Matx33f cam1_rotation_cam1_to_cam0 = cv::Matx33f::eye();
    cv::Vec3f cam1_center_cam0 = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

class GlfwSceneViewer {
public:
    explicit GlfwSceneViewer(GlfwSceneViewerConfig config = {});
    ~GlfwSceneViewer();

    GlfwSceneViewer(const GlfwSceneViewer&) = delete;
    GlfwSceneViewer& operator=(const GlfwSceneViewer&) = delete;

    bool Initialize();
    bool IsOpen() const;
    bool Render(const StereoFusedHandPoseFrame& frame);
    void Shutdown();

private:
    void HandleInput(float dt_seconds);
    void SetupProjection() const;
    void DrawWorldAxes(float axis_length) const;
    void DrawCam0Axes(float axis_length) const;
    void DrawCam0Frustum(float scale) const;
    void DrawCameraAxes(
        const std::array<float, 3>& center_world,
        const std::array<std::array<float, 3>, 3>& axes_world,
        float axis_length,
        const std::array<float, 3>& color_x,
        const std::array<float, 3>& color_y,
        const std::array<float, 3>& color_z) const;
    void DrawCameraFrustum(
        const std::array<float, 3>& center_world,
        const std::array<std::array<float, 3>, 3>& axes_world,
        float scale,
        const std::array<float, 3>& color) const;
    void DrawCam1Axes(float axis_length) const;
    void DrawCam1Frustum(float scale) const;
    void DrawGroundGrid(float extent, float step) const;
    void DrawHands(const StereoFusedHandPoseFrame& frame) const;
    bool LoadManoFaces();

    GlfwSceneViewerConfig config_;
    GLFWwindow* window_ = nullptr;
    double last_frame_time_seconds_ = 0.0;
    std::vector<std::array<int, 3>> mano_faces_;
};

}  // namespace newnewhand
