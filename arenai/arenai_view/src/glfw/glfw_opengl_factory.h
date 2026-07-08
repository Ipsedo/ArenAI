//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GLFW_OPENGL_FACTORY_H
#define ARENAI_GLFW_OPENGL_FACTORY_H

#include <memory>

#include <arenai_view/factory.h>

#include "../opengl/egl_render_context.h"
#include "../opengl/gl_window.h"
#include "../opengl/opengl_factory.h"

namespace arenai::view {

    // Windowed OpenGL backend: an OpenGL backend that also owns an on-screen window
    // and renders to it. Lives in src/glfw because it is created from a concrete
    // GLFW window; it only depends on the abstract AbstractGlWindow itself.
    class GlfwWindowedBackend final : public OpenGlBackend, public AbstractWindowedGraphicBackend {
    public:
        GlfwWindowedBackend(
            std::shared_ptr<EglRenderContext> context, std::shared_ptr<AbstractGlWindow> window);

        std::shared_ptr<AbstractWindow> get_window() override;

        std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) override;

    private:
        std::shared_ptr<AbstractGlWindow> window_;
    };

    class GlfwViewFactory final : public AbstractWindowedViewFactory {
    public:
        std::unique_ptr<AbstractWindowedGraphicBackend>
        make_backend(int window_width, int window_height, const std::string &title) override;
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_OPENGL_FACTORY_H
