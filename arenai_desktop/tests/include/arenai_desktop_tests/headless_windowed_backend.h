//
// Created by samuel on 28/07/2026.
//

#ifndef ARENAI_DESKTOP_TESTS_HEADLESS_WINDOWED_BACKEND_H
#define ARENAI_DESKTOP_TESTS_HEADLESS_WINDOWED_BACKEND_H

#include <memory>
#include <stdexcept>
#include <string>

#include <arenai_view/backend.h>

// Headless stand-in for the GLFW windowed backend: DesktopGameEnvironment only
// touches it for the player view (renderer + drawables), which the end-to-end
// tests discard. The agents' visions keep their real offscreen backend, built
// internally by the environment.

class NoopDrawable final : public view::AbstractDrawable {
public:
    void draw(
        glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
        glm::vec3 camera_pos) override {}
};

class NoopDrawableFactory final : public view::AbstractDrawableFactory {
public:
    std::unique_ptr<view::AbstractDrawable> make_cube_map(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::filesystem::path &pngs_root_path) override {
        return std::make_unique<NoopDrawable>();
    }

    std::unique_ptr<view::AbstractDrawable> make_diffuse(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices, glm::vec4 color) override {
        return std::make_unique<NoopDrawable>();
    }
};

class NoopPlayerRenderer final : public view::AbstractPlayerRenderer {
public:
    void add_drawable(
        const std::string &name, std::unique_ptr<view::AbstractDrawable> drawable) override {}
    void remove_drawable(const std::string &name) override {}

    void draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override {}

    int get_width() const override { return 1; }
    int get_height() const override { return 1; }

    void make_current() const override {}
    void release_current() const override {}

    void add_hud_drawable(std::unique_ptr<view::AbstractHudDrawable> hud_drawable) override {}
    void set_window_size(int new_width, int new_height) override {}
    glm::mat4 last_view_projection() const override { return glm::mat4(1.f); }
};

class HeadlessWindowedBackend final : public view::AbstractWindowedGraphicBackend {
public:
    std::shared_ptr<view::AbstractRenderContext> render_context() override { return nullptr; }

    std::unique_ptr<view::AbstractOffscreenRenderer> make_offscreen_renderer(
        int width, int height, glm::vec3 light_pos,
        const std::shared_ptr<view::AbstractCamera> &camera) override {
        return nullptr;
    }

    std::shared_ptr<view::AbstractDrawableFactory> drawable_factory() override {
        return std::make_shared<NoopDrawableFactory>();
    }

    std::shared_ptr<view::AbstractHudFactory> hud_factory() override { return nullptr; }

    std::string renderer_info() override { return "headless test backend"; }

    void release_thread() override {}

    std::shared_ptr<view::AbstractWindow> get_window() override { return nullptr; }

    std::unique_ptr<view::AbstractPlayerRenderer> make_player_renderer(
        glm::vec3 light_pos, const std::shared_ptr<view::AbstractCamera> &camera,
        const view::PlayerRendererSettings &settings) override {
        return std::make_unique<NoopPlayerRenderer>();
    }

    Rml::RenderInterface &ui_render_interface() override {
        throw std::logic_error("no UI in the headless test backend");
    }
    void begin_ui_frame(int width, int height) override {}
    void begin_ui_overlay(int width, int height) override {}
    void end_ui_frame() override {}
    void present() override {}
};

#endif// ARENAI_DESKTOP_TESTS_HEADLESS_WINDOWED_BACKEND_H
