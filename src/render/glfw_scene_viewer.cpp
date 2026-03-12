#include "newnewhand/render/glfw_scene_viewer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
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

void DrawTextBitmap(float x, float y, const std::string& text) {
    glRasterPos2f(x, y);
    for (char c : text) {
        glfwGetProcAddress("glutBitmapCharacter");
        (void)c;
    }
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

    if (!LoadManoFaces()) {
        std::cerr << "[glfw] failed to load MANO faces from resources\n";
    }

    last_frame_time_seconds_ = glfwGetTime();
    std::cerr << "[glfw] controls: W/S pitch, A/D yaw, Q/E roll, ESC close window\n";
    return true;
}

bool GlfwSceneViewer::IsOpen() const {
    return window_ != nullptr && glfwWindowShouldClose(window_) == GLFW_FALSE;
}

bool GlfwSceneViewer::Render(const StereoFusedHandPoseFrame& frame) {
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

    if (config_.draw_world_axes || config_.draw_cam0_axes) {
        DrawAxes(config_.axis_length);
    }
    if (config_.draw_cam0_frustum) {
        DrawCam0Frustum(config_.axis_length * 0.7f);
    }
    if (config_.draw_mesh) {
        DrawHands(frame);
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
        config_.pitch_degrees += delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        config_.pitch_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        config_.yaw_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        config_.yaw_degrees += delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) {
        config_.roll_degrees -= delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) {
        config_.roll_degrees += delta;
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

void GlfwSceneViewer::DrawAxes(float axis_length) const {
    DrawLine({0.0f, 0.0f, 0.0f}, {axis_length, 0.0f, 0.0f}, {1.0f, 0.15f, 0.15f});
    DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, axis_length, 0.0f}, {0.15f, 1.0f, 0.15f});
    DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, axis_length}, {0.15f, 0.45f, 1.0f});
}

void GlfwSceneViewer::DrawCam0Frustum(float scale) const {
    const std::array<std::array<float, 3>, 4> plane = {{
        {-0.6f * scale, -0.4f * scale, scale},
        {0.6f * scale, -0.4f * scale, scale},
        {0.6f * scale, 0.4f * scale, scale},
        {-0.6f * scale, 0.4f * scale, scale},
    }};

    for (const auto& corner : plane) {
        DrawLine({0.0f, 0.0f, 0.0f}, corner, {0.80f, 0.80f, 0.15f});
    }
    for (std::size_t i = 0; i < plane.size(); ++i) {
        DrawLine(plane[i], plane[(i + 1) % plane.size()], {0.80f, 0.80f, 0.15f});
    }
}

void GlfwSceneViewer::DrawHands(const StereoFusedHandPoseFrame& frame) const {
    if (mano_faces_.empty()) {
        return;
    }

    for (const auto& hand : frame.hands) {
        const std::array<float, 3> base = hand.is_right
            ? std::array<float, 3>{0.95f, 0.70f, 0.20f}
            : std::array<float, 3>{0.82f, 0.35f, 0.95f};

        glColor4f(base[0], base[1], base[2], 0.78f);
        glBegin(GL_TRIANGLES);
        for (const auto& face : mano_faces_) {
            const float* v0 = hand.pose_cam0.vertices[face[0]];
            const float* v1 = hand.pose_cam0.vertices[face[1]];
            const float* v2 = hand.pose_cam0.vertices[face[2]];
            glVertex3f(v0[0] + hand.pose_cam0.camera_translation[0], v0[1] + hand.pose_cam0.camera_translation[1], v0[2] + hand.pose_cam0.camera_translation[2]);
            glVertex3f(v1[0] + hand.pose_cam0.camera_translation[0], v1[1] + hand.pose_cam0.camera_translation[1], v1[2] + hand.pose_cam0.camera_translation[2]);
            glVertex3f(v2[0] + hand.pose_cam0.camera_translation[0], v2[1] + hand.pose_cam0.camera_translation[1], v2[2] + hand.pose_cam0.camera_translation[2]);
        }
        glEnd();

        if (config_.draw_wireframe) {
            glColor3f(0.05f, 0.05f, 0.05f);
            glBegin(GL_LINES);
            for (const auto& face : mano_faces_) {
                const float* v0 = hand.pose_cam0.vertices[face[0]];
                const float* v1 = hand.pose_cam0.vertices[face[1]];
                const float* v2 = hand.pose_cam0.vertices[face[2]];
                const std::array<std::array<const float*, 2>, 3> edges = {{{v0, v1}, {v1, v2}, {v2, v0}}};
                for (const auto& edge : edges) {
                    glVertex3f(edge[0][0] + hand.pose_cam0.camera_translation[0], edge[0][1] + hand.pose_cam0.camera_translation[1], edge[0][2] + hand.pose_cam0.camera_translation[2]);
                    glVertex3f(edge[1][0] + hand.pose_cam0.camera_translation[0], edge[1][1] + hand.pose_cam0.camera_translation[1], edge[1][2] + hand.pose_cam0.camera_translation[2]);
                }
            }
            glEnd();
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

void GlfwSceneViewer::DrawOverlayText() const {
}

}  // namespace newnewhand
