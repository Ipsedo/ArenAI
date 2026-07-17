//
// Created by samuel on 08/07/2026.
//

#include "./glfw_opengl_backend.h"

#include <utility>

#include <arenai_view/backend.h>

#include "../opengl/renderers/gl_renderer.h"
#include "./glfw_window.h"

namespace arenai::view {

    /*
     * OpenGlWindowedBackend
     */

    GlfwWindowedBackend::GlfwWindowedBackend(
        std::shared_ptr<EglRenderContext> context, std::shared_ptr<AbstractGlWindow> window,
        const int window_width, const int window_height)
        : OpenGlBackend(std::move(context)), window_(std::move(window)),
          window_width_(window_width), window_height_(window_height) {}

    std::shared_ptr<AbstractWindow> GlfwWindowedBackend::get_window() { return window_; }

    std::unique_ptr<AbstractPlayerRenderer> GlfwWindowedBackend::make_player_renderer(
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera) {
        return std::make_unique<GlPlayerRenderer>(
            context_, window_width_, window_height_, light_pos, camera);
    }

    Rml::RenderInterface &GlfwWindowedBackend::ui_render_interface() {
        return rml_render_interface_;
    }

    void GlfwWindowedBackend::begin_ui_frame(const int width, const int height) {
        context_->make_current();

        glViewport(0, 0, width, height);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        rml_render_interface_.begin_frame(width, height);
    }

    void GlfwWindowedBackend::begin_ui_overlay(const int width, const int height) {
        context_->make_current();

        // no clear: the UI is composited over the frame already drawn
        glViewport(0, 0, width, height);

        rml_render_interface_.begin_frame(width, height);
    }

    void GlfwWindowedBackend::end_ui_frame() { rml_render_interface_.end_frame(); }

    void GlfwWindowedBackend::present() {
        eglSwapBuffers(context_->get_display(), context_->get_surface());
    }

    /*
     * OpenGlViewFactory: windowed backend construction (GLFW-specific).
     */

    std::unique_ptr<AbstractWindowedGraphicBackend>
    make_glfw_backend(const int window_width, const int window_height, const std::string &title) {
        auto window = std::make_shared<GlfwWindow>(window_width, window_height, title);
        auto context = std::make_shared<NativeEglContext>(
            window->egl_display(), window->egl_surface(), window->egl_context());
        return std::make_unique<GlfwWindowedBackend>(
            std::move(context), std::move(window), window_width, window_height);
    }

}// namespace arenai::view
