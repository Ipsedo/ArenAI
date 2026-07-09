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
