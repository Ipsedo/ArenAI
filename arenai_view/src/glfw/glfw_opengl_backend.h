//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GLFW_OPENGL_FACTORY_H
#define ARENAI_GLFW_OPENGL_FACTORY_H

#include <memory>

#include <arenai_view/backend.h>

#include "../opengl/egl_render_context.h"
#include "../opengl/gl_window.h"
#include "../opengl/opengl_backend.h"
#include "../opengl/rml_render_interface.h"

namespace arenai::view {

    class GlfwWindowedBackend final : public OpenGlBackend, public AbstractWindowedGraphicBackend {
    public:
        GlfwWindowedBackend(
            std::shared_ptr<EglRenderContext> context, std::shared_ptr<AbstractGlWindow> window,
            int window_width, int window_height);

        std::shared_ptr<AbstractWindow> get_window() override;

        std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera) override;

        Rml::RenderInterface &ui_render_interface() override;
        void begin_ui_frame(int width, int height) override;
        void begin_ui_overlay(int width, int height) override;
        void end_ui_frame() override;
        void present() override;

    private:
        std::shared_ptr<AbstractGlWindow> window_;

        int window_width_;
        int window_height_;

        RmlGlRenderInterface rml_render_interface_;
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_OPENGL_FACTORY_H
