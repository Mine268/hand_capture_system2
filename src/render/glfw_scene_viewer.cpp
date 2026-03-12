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
    std::cerr << "[glfw] controls: W/S pitch, A/D yaw, Q/E roll, I/K pan Y, J/L pan X, Z/X zoom, ESC close window\n";
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
    glTranslatef(config_.pan_x, config_.pan_y, 0.0f);
    glRotatef(config_.roll_degrees, 0.0f, 0.0f, 1.0f);
    glRotatef(config_.pitch_degrees, 1.0f, 0.0f, 0.0f);
    glRotatef(config_.yaw_degrees, 0.0f, 1.0f, 0.0f);

    if (config_.draw_ground_grid) {
        DrawGroundGrid(0.35f, 0.05f);
    }
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
    if (glfwGetKey(window_, GLFW_KEY_J) == GLFW_PRESS) {
        config_.pan_x += config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_L) == GLFW_PRESS) {
        config_.pan_x -= config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_I) == GLFW_PRESS) {
        config_.pan_y -= config_.translation_speed * dt_seconds;
    }
    if (glfwGetKey(window_, GLFW_KEY_K) == GLFW_PRESS) {
        config_.pan_y += config_.translation_speed * dt_seconds;
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

void GlfwSceneViewer::DrawAxes(float axis_length) const {
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

void GlfwSceneViewer::DrawCam0Frustum(float scale) const {
    glDisable(GL_LIGHTING);
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

void GlfwSceneViewer::DrawHands(const StereoFusedHandPoseFrame& frame) const {
    if (mano_faces_.empty()) {
        return;
    }

    for (const auto& hand : frame.hands) {
        const std::array<float, 3> base = hand.is_right
            ? std::array<float, 3>{0.95f, 0.70f, 0.20f}
            : std::array<float, 3>{0.82f, 0.35f, 0.95f};

        std::array<std::array<float, 3>, 778> vertices{};
        std::array<std::array<float, 3>, 778> normals{};
        for (int i = 0; i < 778; ++i) {
            vertices[i] = {
                hand.pose_cam0.vertices[i][0] + hand.pose_cam0.camera_translation[0],
                hand.pose_cam0.vertices[i][1] + hand.pose_cam0.camera_translation[1],
                hand.pose_cam0.vertices[i][2] + hand.pose_cam0.camera_translation[2],
            };
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
