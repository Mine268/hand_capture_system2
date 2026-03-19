#include "newnewhand/render/glfw_scene_viewer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace newnewhand {

namespace {

std::string DefaultManoFacePath() {
#ifdef NEWNEWHAND_PROJECT_ROOT
    return std::string(NEWNEWHAND_PROJECT_ROOT) + "/resources/mano_faces.txt";
#else
    return {};
#endif
}

void DrawLine(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& color) {
    glColor3f(color[0], color[1], color[2]);
    glBegin(GL_LINES);
    glVertex3f(a[0], a[1], a[2]);
    glVertex3f(b[0], b[1], b[2]);
    glEnd();
}

void DrawAxisLabelX(const std::array<float, 3>& origin, float scale, const std::array<float, 3>& color) {
    DrawLine(
        {origin[0] - scale, origin[1] - scale, origin[2]},
        {origin[0] + scale, origin[1] + scale, origin[2]},
        color);
    DrawLine(
        {origin[0] - scale, origin[1] + scale, origin[2]},
        {origin[0] + scale, origin[1] - scale, origin[2]},
        color);
}

void DrawAxisLabelY(const std::array<float, 3>& origin, float scale, const std::array<float, 3>& color) {
    DrawLine(
        {origin[0] - scale, origin[1] + scale, origin[2]},
        {origin[0], origin[1], origin[2]},
        color);
    DrawLine(
        {origin[0] + scale, origin[1] + scale, origin[2]},
        {origin[0], origin[1], origin[2]},
        color);
    DrawLine(
        {origin[0], origin[1], origin[2]},
        {origin[0], origin[1] - scale * 1.4f, origin[2]},
        color);
}

void DrawAxisLabelZ(const std::array<float, 3>& origin, float scale, const std::array<float, 3>& color) {
    DrawLine(
        {origin[0] - scale, origin[1] + scale, origin[2]},
        {origin[0] + scale, origin[1] + scale, origin[2]},
        color);
    DrawLine(
        {origin[0] + scale, origin[1] + scale, origin[2]},
        {origin[0] - scale, origin[1] - scale, origin[2]},
        color);
    DrawLine(
        {origin[0] - scale, origin[1] - scale, origin[2]},
        {origin[0] + scale, origin[1] - scale, origin[2]},
        color);
}

std::array<float, 3> Cross(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

void Normalize(std::array<float, 3>& v) {
    const float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm > 1e-6f) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

std::array<float, 3> Sub(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

std::array<float, 3> TransformCam0ToWorld(const std::array<float, 3>& p) {
    return {-p[0], -p[1], p[2]};
}

std::array<float, 3> TransformDirectionCam0ToWorld(const std::array<float, 3>& d) {
    return {-d[0], -d[1], d[2]};
}

std::array<float, 3> TransformSlamWorldToViewerWorld(const std::array<float, 3>& p) {
    return {-p[0], -p[1], p[2]};
}

std::array<float, 3> TransformSlamDirectionToViewerWorld(const std::array<float, 3>& d) {
    return {-d[0], -d[1], d[2]};
}

std::array<float, 3> MatVecMul(const cv::Matx33f& matrix, const std::array<float, 3>& v) {
    return {
        matrix(0, 0) * v[0] + matrix(0, 1) * v[1] + matrix(0, 2) * v[2],
        matrix(1, 0) * v[0] + matrix(1, 1) * v[1] + matrix(1, 2) * v[2],
        matrix(2, 0) * v[0] + matrix(2, 1) * v[1] + matrix(2, 2) * v[2],
    };
}

cv::Vec3f MatVecMul(const cv::Matx33f& matrix, const cv::Vec3f& v) {
    return cv::Vec3f(
        matrix(0, 0) * v[0] + matrix(0, 1) * v[1] + matrix(0, 2) * v[2],
        matrix(1, 0) * v[0] + matrix(1, 1) * v[1] + matrix(1, 2) * v[2],
        matrix(2, 0) * v[0] + matrix(2, 1) * v[1] + matrix(2, 2) * v[2]);
}

std::array<float, 3> ToArray(const cv::Vec3f& v) {
    return {v[0], v[1], v[2]};
}

}  // namespace

GlfwSceneViewer::GlfwSceneViewer(GlfwSceneViewerConfig config)
    : config_(std::move(config)) {}

GlfwSceneViewer::~GlfwSceneViewer() {
    Shutdown();
}

bool GlfwSceneViewer::Initialize() {
    if (window_) {
        return true;
    }

    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "[glfw] failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    window_ = glfwCreateWindow(config_.width, config_.height, config_.title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "[glfw] failed to create OpenGL window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    const GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        std::cerr << "[glfw] glewInit failed: " << glewGetErrorString(glew_status) << "\n";
        Shutdown();
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    const GLfloat light0_position[] = {0.25f, 0.85f, 1.1f, 0.0f};
    const GLfloat light0_diffuse[] = {0.95f, 0.95f, 0.92f, 1.0f};
    const GLfloat light0_specular[] = {0.35f, 0.35f, 0.35f, 1.0f};
    const GLfloat light1_position[] = {-0.8f, -0.3f, 0.6f, 0.0f};
    const GLfloat light1_diffuse[] = {0.32f, 0.36f, 0.48f, 1.0f};
    const GLfloat global_ambient[] = {0.18f, 0.18f, 0.18f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    if (!LoadManoFaces()) {
        std::cerr << "[glfw] failed to load MANO faces from resources\n";
    }

    last_frame_time_seconds_ = glfwGetTime();
    std::cerr << "[glfw] controls: W/A/S/D move in world XZ, Q/E move in world Y, arrow keys rotate yaw/pitch, U/O roll, Z/X zoom, ESC close window\n";
    return true;
}

bool GlfwSceneViewer::IsOpen() const {
    return window_ != nullptr && glfwWindowShouldClose(window_) == GLFW_FALSE;
}

void GlfwSceneViewer::SetTitle(const std::string& title) {
    config_.title = title;
    if (window_) {
        glfwSetWindowTitle(window_, config_.title.c_str());
    }
}

bool GlfwSceneViewer::Render(
    const StereoFusedHandPoseFrame& frame,
    const StereoCameraTrackingResult* tracking) {
    if (!window_ && !Initialize()) {
        return false;
    }
    if (!IsOpen()) {
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwPollEvents();

    const double now_seconds = glfwGetTime();
    const float dt_seconds = static_cast<float>(now_seconds - last_frame_time_seconds_);
    last_frame_time_seconds_ = now_seconds;
    HandleInput(dt_seconds);

    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
    glViewport(0, 0, framebuffer_width, framebuffer_height);

    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SetupProjection();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -config_.camera_distance);
    glRotatef(config_.roll_degrees, 0.0f, 0.0f, 1.0f);
    glRotatef(config_.pitch_degrees, 1.0f, 0.0f, 0.0f);
    glRotatef(config_.yaw_degrees, 0.0f, 1.0f, 0.0f);
    glTranslatef(-config_.world_offset_x, -config_.world_offset_y, -config_.world_offset_z);

    const bool use_tracked_world = tracking != nullptr && tracking->initialized;

    if (config_.draw_ground_grid) {
        DrawGroundGrid(0.35f, 0.05f);
    }
    if (config_.draw_world_axes) {
        DrawWorldAxes(config_.axis_length);
    }
    if (!use_tracked_world && config_.draw_cam0_axes) {
        DrawCam0Axes(config_.axis_length * 0.85f);
    }
    if (!use_tracked_world && config_.draw_cam0_frustum) {
        DrawCam0Frustum(config_.axis_length * 0.7f);
    }
    if (!use_tracked_world && config_.draw_cam1_frustum) {
        DrawCam1Frustum(config_.axis_length * 0.7f);
    }
    if (!use_tracked_world && config_.draw_cam1_axes) {
        DrawCam1Axes(config_.axis_length * 0.85f);
    }
    if (use_tracked_world && config_.draw_slam_origin_axes) {
        DrawSlamOriginAxes(config_.axis_length);
    }
    if (use_tracked_world && config_.draw_cam0_axes) {
        DrawTrackedCam0Axes(*tracking, config_.axis_length * 0.85f);
    }
    if (use_tracked_world && config_.draw_cam0_frustum) {
        DrawTrackedCam0Frustum(*tracking, config_.axis_length * 0.7f);
    }
    if (use_tracked_world && config_.draw_cam1_axes) {
        DrawTrackedCam1Axes(*tracking, config_.axis_length * 0.85f);
    }
    if (use_tracked_world && config_.draw_cam1_frustum) {
        DrawTrackedCam1Frustum(*tracking, config_.axis_length * 0.7f);
    }
    if (use_tracked_world && config_.draw_slam_trajectory) {
        DrawSlamTrajectory(*tracking);
    }
    if (config_.draw_mesh) {
        DrawHands(frame, tracking);
    }

    glfwSwapBuffers(window_);
    return true;
}

void GlfwSceneViewer::Shutdown() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void GlfwSceneViewer::HandleInput(float dt_seconds) {
    const float delta = config_.angular_speed_degrees * dt_seconds;
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
        config_.world_offset_z += config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        config_.world_offset_z -= config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        config_.world_offset_x -= config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        config_.world_offset_x += config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) {
        config_.world_offset_y += config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) {
        config_.world_offset_y -= config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS) {
        config_.yaw_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        config_.yaw_degrees += delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS) {
        config_.pitch_degrees += delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS) {
        config_.pitch_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_U) == GLFW_PRESS) {
        config_.roll_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_O) == GLFW_PRESS) {
        config_.roll_degrees += delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_Z) == GLFW_PRESS) {
        config_.camera_distance = std::max(0.08f, config_.camera_distance - config_.zoom_speed * dt_seconds);
    }
    if (glfwGetKey(window_, GLFW_KEY_X) == GLFW_PRESS) {
        config_.camera_distance += config_.zoom_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
}

void GlfwSceneViewer::SetupProjection() const {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const float aspect = static_cast<float>(config_.width) / static_cast<float>(config_.height);
    constexpr float fov_y_degrees = 45.0f;
    constexpr float near_plane = 0.01f;
    constexpr float far_plane = 20.0f;
    const float top = std::tan(fov_y_degrees * 0.5f * static_cast<float>(M_PI) / 180.0f) * near_plane;
    const float right = top * aspect;
    glFrustum(-right, right, -top, top, near_plane, far_plane);
}

void GlfwSceneViewer::DrawWorldAxes(float axis_length) const {
    glDisable(GL_LIGHTING);
    glLineWidth(2.5f);
    DrawLine({0.0f, 0.0f, 0.0f}, {axis_length, 0.0f, 0.0f}, {1.0f, 0.15f, 0.15f});
    DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, axis_length, 0.0f}, {0.15f, 1.0f, 0.15f});
    DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, axis_length}, {0.15f, 0.45f, 1.0f});

    const float label_scale = axis_length * 0.12f;
    DrawAxisLabelX({axis_length * 1.15f, 0.0f, 0.0f}, label_scale, {1.0f, 0.15f, 0.15f});
    DrawAxisLabelY({0.0f, axis_length * 1.15f, 0.0f}, label_scale, {0.15f, 1.0f, 0.15f});
    DrawAxisLabelZ({0.0f, 0.0f, axis_length * 1.15f}, label_scale, {0.15f, 0.45f, 1.0f});

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void GlfwSceneViewer::DrawCam0Axes(float axis_length) const {
    DrawCameraAxes(
        {0.0f, 0.0f, 0.0f},
        {{
            TransformDirectionCam0ToWorld({1.0f, 0.0f, 0.0f}),
            TransformDirectionCam0ToWorld({0.0f, 1.0f, 0.0f}),
            TransformDirectionCam0ToWorld({0.0f, 0.0f, 1.0f}),
        }},
        axis_length,
        {1.0f, 0.70f, 0.25f},
        {0.45f, 1.0f, 0.45f},
        {0.35f, 0.75f, 1.0f});
}

void GlfwSceneViewer::DrawCameraAxes(
    const std::array<float, 3>& center_world,
    const std::array<std::array<float, 3>, 3>& axes_world,
    float axis_length,
    const std::array<float, 3>& color_x,
    const std::array<float, 3>& color_y,
    const std::array<float, 3>& color_z) const {
    glDisable(GL_LIGHTING);
    glLineWidth(1.5f);
    DrawLine(center_world, {
        center_world[0] + axes_world[0][0] * axis_length,
        center_world[1] + axes_world[0][1] * axis_length,
        center_world[2] + axes_world[0][2] * axis_length,
    }, color_x);
    DrawLine(center_world, {
        center_world[0] + axes_world[1][0] * axis_length,
        center_world[1] + axes_world[1][1] * axis_length,
        center_world[2] + axes_world[1][2] * axis_length,
    }, color_y);
    DrawLine(center_world, {
        center_world[0] + axes_world[2][0] * axis_length,
        center_world[1] + axes_world[2][1] * axis_length,
        center_world[2] + axes_world[2][2] * axis_length,
    }, color_z);
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void GlfwSceneViewer::DrawCam0Frustum(float scale) const {
    DrawCameraFrustum(
        {0.0f, 0.0f, 0.0f},
        {{
            TransformDirectionCam0ToWorld({1.0f, 0.0f, 0.0f}),
            TransformDirectionCam0ToWorld({0.0f, 1.0f, 0.0f}),
            TransformDirectionCam0ToWorld({0.0f, 0.0f, 1.0f}),
        }},
        scale,
        {0.80f, 0.80f, 0.15f});
}

void GlfwSceneViewer::DrawCameraFrustum(
    const std::array<float, 3>& center_world,
    const std::array<std::array<float, 3>, 3>& axes_world,
    float scale,
    const std::array<float, 3>& color) const {
    glDisable(GL_LIGHTING);
    const std::array<std::array<float, 3>, 4> plane = {{
        {-0.6f * scale, -0.4f * scale, scale},
        {0.6f * scale, -0.4f * scale, scale},
        {0.6f * scale, 0.4f * scale, scale},
        {-0.6f * scale, 0.4f * scale, scale},
    }};

    for (const auto& corner : plane) {
        const std::array<float, 3> world_corner = {
            center_world[0] + axes_world[0][0] * corner[0] + axes_world[1][0] * corner[1] + axes_world[2][0] * corner[2],
            center_world[1] + axes_world[0][1] * corner[0] + axes_world[1][1] * corner[1] + axes_world[2][1] * corner[2],
            center_world[2] + axes_world[0][2] * corner[0] + axes_world[1][2] * corner[1] + axes_world[2][2] * corner[2],
        };
        DrawLine(center_world, world_corner, color);
    }
    for (std::size_t i = 0; i < plane.size(); ++i) {
        const auto& a = plane[i];
        const auto& b = plane[(i + 1) % plane.size()];
        const std::array<float, 3> world_a = {
            center_world[0] + axes_world[0][0] * a[0] + axes_world[1][0] * a[1] + axes_world[2][0] * a[2],
            center_world[1] + axes_world[0][1] * a[0] + axes_world[1][1] * a[1] + axes_world[2][1] * a[2],
            center_world[2] + axes_world[0][2] * a[0] + axes_world[1][2] * a[1] + axes_world[2][2] * a[2],
        };
        const std::array<float, 3> world_b = {
            center_world[0] + axes_world[0][0] * b[0] + axes_world[1][0] * b[1] + axes_world[2][0] * b[2],
            center_world[1] + axes_world[0][1] * b[0] + axes_world[1][1] * b[1] + axes_world[2][1] * b[2],
            center_world[2] + axes_world[0][2] * b[0] + axes_world[1][2] * b[1] + axes_world[2][2] * b[2],
        };
        DrawLine(world_a, world_b, color);
    }
    glEnable(GL_LIGHTING);
}

void GlfwSceneViewer::DrawCam1Axes(float axis_length) const {
    if (!config_.has_cam1_pose) {
        return;
    }
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{1.0f, 0.0f, 0.0f})),
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{0.0f, 1.0f, 0.0f})),
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{0.0f, 0.0f, 1.0f})),
    }};
    DrawCameraAxes(
        TransformCam0ToWorld({config_.cam1_center_cam0[0], config_.cam1_center_cam0[1], config_.cam1_center_cam0[2]}),
        axes_world,
        axis_length,
        {0.85f, 0.35f, 0.35f},
        {0.35f, 0.90f, 0.35f},
        {0.35f, 0.60f, 0.95f});
}

void GlfwSceneViewer::DrawCam1Frustum(float scale) const {
    if (!config_.has_cam1_pose) {
        return;
    }
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{1.0f, 0.0f, 0.0f})),
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{0.0f, 1.0f, 0.0f})),
        TransformDirectionCam0ToWorld(MatVecMul(config_.cam1_rotation_cam1_to_cam0, std::array<float, 3>{0.0f, 0.0f, 1.0f})),
    }};
    DrawCameraFrustum(
        TransformCam0ToWorld({config_.cam1_center_cam0[0], config_.cam1_center_cam0[1], config_.cam1_center_cam0[2]}),
        axes_world,
        scale,
        {0.35f, 0.85f, 0.95f});
}

void GlfwSceneViewer::DrawTrackedCam0Axes(
    const StereoCameraTrackingResult& tracking,
    float axis_length) const {
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(1.0f, 0.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(0.0f, 1.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(0.0f, 0.0f, 1.0f)))),
    }};
    DrawCameraAxes(
        TransformSlamWorldToViewerWorld(ToArray(tracking.camera_center_world)),
        axes_world,
        axis_length,
        {1.0f, 0.70f, 0.25f},
        {0.45f, 1.0f, 0.45f},
        {0.35f, 0.75f, 1.0f});
}

void GlfwSceneViewer::DrawTrackedCam0Frustum(
    const StereoCameraTrackingResult& tracking,
    float scale) const {
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(1.0f, 0.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(0.0f, 1.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(tracking.rotation_world_from_cam0, cv::Vec3f(0.0f, 0.0f, 1.0f)))),
    }};
    DrawCameraFrustum(
        TransformSlamWorldToViewerWorld(ToArray(tracking.camera_center_world)),
        axes_world,
        scale,
        {0.80f, 0.80f, 0.15f});
}

void GlfwSceneViewer::DrawTrackedCam1Axes(
    const StereoCameraTrackingResult& tracking,
    float axis_length) const {
    if (!config_.has_cam1_pose) {
        return;
    }
    const cv::Matx33f rotation_world_from_cam1 =
        tracking.rotation_world_from_cam0 * config_.cam1_rotation_cam1_to_cam0;
    const cv::Vec3f cam1_center_world =
        MatVecMul(tracking.rotation_world_from_cam0, config_.cam1_center_cam0)
        + tracking.camera_center_world;
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(1.0f, 0.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(0.0f, 1.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(0.0f, 0.0f, 1.0f)))),
    }};
    DrawCameraAxes(
        TransformSlamWorldToViewerWorld(ToArray(cam1_center_world)),
        axes_world,
        axis_length,
        {0.85f, 0.35f, 0.35f},
        {0.35f, 0.90f, 0.35f},
        {0.35f, 0.60f, 0.95f});
}

void GlfwSceneViewer::DrawTrackedCam1Frustum(
    const StereoCameraTrackingResult& tracking,
    float scale) const {
    if (!config_.has_cam1_pose) {
        return;
    }
    const cv::Matx33f rotation_world_from_cam1 =
        tracking.rotation_world_from_cam0 * config_.cam1_rotation_cam1_to_cam0;
    const cv::Vec3f cam1_center_world =
        MatVecMul(tracking.rotation_world_from_cam0, config_.cam1_center_cam0)
        + tracking.camera_center_world;
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(1.0f, 0.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(0.0f, 1.0f, 0.0f)))),
        TransformSlamDirectionToViewerWorld(ToArray(MatVecMul(rotation_world_from_cam1, cv::Vec3f(0.0f, 0.0f, 1.0f)))),
    }};
    DrawCameraFrustum(
        TransformSlamWorldToViewerWorld(ToArray(cam1_center_world)),
        axes_world,
        scale,
        {0.35f, 0.85f, 0.95f});
}

void GlfwSceneViewer::DrawSlamOriginAxes(float axis_length) const {
    const std::array<std::array<float, 3>, 3> axes_world = {{
        TransformSlamDirectionToViewerWorld({1.0f, 0.0f, 0.0f}),
        TransformSlamDirectionToViewerWorld({0.0f, 1.0f, 0.0f}),
        TransformSlamDirectionToViewerWorld({0.0f, 0.0f, 1.0f}),
    }};
    DrawCameraAxes(
        TransformSlamWorldToViewerWorld({0.0f, 0.0f, 0.0f}),
        axes_world,
        axis_length,
        {0.95f, 0.25f, 0.25f},
        {0.25f, 0.95f, 0.25f},
        {0.25f, 0.60f, 1.0f});
}

void GlfwSceneViewer::DrawSlamTrajectory(const StereoCameraTrackingResult& tracking) const {
    if (tracking.trajectory_world.size() < 2) {
        return;
    }
    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.85f, 0.20f);
    glBegin(GL_LINE_STRIP);
    for (const auto& center_world : tracking.trajectory_world) {
        const auto point_viewer = TransformSlamWorldToViewerWorld(ToArray(center_world));
        glVertex3f(point_viewer[0], point_viewer[1], point_viewer[2]);
    }
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void GlfwSceneViewer::DrawGroundGrid(float extent, float step) const {
    glDisable(GL_LIGHTING);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (float v = -extent; v <= extent + 1e-6f; v += step) {
        const bool major = std::abs(std::fmod(v + extent, step * 4.0f)) < 1e-4f;
        const float color = major ? 0.28f : 0.18f;
        glColor3f(color, color, color);
        glVertex3f(v, -extent, 0.0f);
        glVertex3f(v, extent, 0.0f);
        glVertex3f(-extent, v, 0.0f);
        glVertex3f(extent, v, 0.0f);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void GlfwSceneViewer::DrawHands(
    const StereoFusedHandPoseFrame& frame,
    const StereoCameraTrackingResult* tracking) const {
    if (mano_faces_.empty()) {
        return;
    }

    const bool use_tracked_world = tracking != nullptr && tracking->initialized;
    for (const auto& hand : frame.hands) {
        const std::array<float, 3> base = hand.is_right
            ? std::array<float, 3>{0.95f, 0.70f, 0.20f}
            : std::array<float, 3>{0.82f, 0.35f, 0.95f};

        std::array<std::array<float, 3>, 778> vertices{};
        std::array<std::array<float, 3>, 778> normals{};
        for (int i = 0; i < 778; ++i) {
            const std::array<float, 3> cam_vertex = {
                hand.pose_cam0.vertices[i][0] + hand.pose_cam0.camera_translation[0],
                hand.pose_cam0.vertices[i][1] + hand.pose_cam0.camera_translation[1],
                hand.pose_cam0.vertices[i][2] + hand.pose_cam0.camera_translation[2],
            };
            if (use_tracked_world) {
                const cv::Vec3f vertex_world =
                    MatVecMul(tracking->rotation_world_from_cam0, cv::Vec3f(cam_vertex[0], cam_vertex[1], cam_vertex[2]))
                    + tracking->camera_center_world;
                vertices[i] = TransformSlamWorldToViewerWorld(ToArray(vertex_world));
            } else {
                vertices[i] = TransformCam0ToWorld(cam_vertex);
            }
            normals[i] = {0.0f, 0.0f, 0.0f};
        }
        for (const auto& face : mano_faces_) {
            const auto e1 = Sub(vertices[face[1]], vertices[face[0]]);
            const auto e2 = Sub(vertices[face[2]], vertices[face[0]]);
            auto n = Cross(e1, e2);
            Normalize(n);
            for (int idx : face) {
                normals[idx][0] += n[0];
                normals[idx][1] += n[1];
                normals[idx][2] += n[2];
            }
        }
        for (auto& n : normals) {
            Normalize(n);
        }

        const GLfloat specular[] = {0.25f, 0.25f, 0.25f, 1.0f};
        const GLfloat shininess[] = {32.0f};
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
        glDisable(GL_BLEND);
        glColor4f(base[0], base[1], base[2], 1.0f);
        glBegin(GL_TRIANGLES);
        for (const auto& face : mano_faces_) {
            for (int corner = 0; corner < 3; ++corner) {
                const int idx = face[corner];
                glNormal3f(normals[idx][0], normals[idx][1], normals[idx][2]);
                glVertex3f(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
            }
        }
        glEnd();
        glEnable(GL_BLEND);

        if (config_.draw_wireframe) {
            glDisable(GL_LIGHTING);
            glColor3f(0.05f, 0.05f, 0.05f);
            glBegin(GL_LINES);
            for (const auto& face : mano_faces_) {
                const auto& v0 = vertices[face[0]];
                const auto& v1 = vertices[face[1]];
                const auto& v2 = vertices[face[2]];
                const std::array<std::array<std::array<float, 3>, 2>, 3> edges = {{{v0, v1}, {v1, v2}, {v2, v0}}};
                for (const auto& edge : edges) {
                    glVertex3f(edge[0][0], edge[0][1], edge[0][2]);
                    glVertex3f(edge[1][0], edge[1][1], edge[1][2]);
                }
            }
            glEnd();
            glEnable(GL_LIGHTING);
        }
    }
}

bool GlfwSceneViewer::LoadManoFaces() {
    const std::string path = DefaultManoFacePath();
    if (path.empty()) {
        return false;
    }
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }
    mano_faces_.clear();
    std::array<int, 3> face{};
    while (input >> face[0] >> face[1] >> face[2]) {
        mano_faces_.push_back(face);
    }
    return !mano_faces_.empty();
}
}  // namespace newnewhand
