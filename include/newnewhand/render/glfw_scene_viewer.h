#pragma once

#include <array>
#include <string>
#include <vector>

#include "newnewhand/fusion/stereo_hand_fuser.h"

struct GLFWwindow;

namespace newnewhand {

struct GlfwSceneViewerConfig {
    int width = 1280;
    int height = 900;
    std::string title = "newnewhand OpenGL Viewer";
    float camera_distance = 0.6f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    float yaw_degrees = -35.0f;
    float pitch_degrees = 18.0f;
    float roll_degrees = 0.0f;
    float angular_speed_degrees = 70.0f;
    float translation_speed = 0.25f;
    float zoom_speed = 0.6f;
    float axis_length = 0.10f;
    bool draw_world_axes = true;
    bool draw_cam0_axes = true;
    bool draw_cam0_frustum = true;
    bool draw_ground_grid = true;
    bool draw_mesh = true;
    bool draw_wireframe = false;
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
    void DrawAxes(float axis_length) const;
    void DrawCam0Frustum(float scale) const;
    void DrawGroundGrid(float extent, float step) const;
    void DrawHands(const StereoFusedHandPoseFrame& frame) const;
    bool LoadManoFaces();

    GlfwSceneViewerConfig config_;
    GLFWwindow* window_ = nullptr;
    double last_frame_time_seconds_ = 0.0;
    std::vector<std::array<int, 3>> mano_faces_;
};

}  // namespace newnewhand
