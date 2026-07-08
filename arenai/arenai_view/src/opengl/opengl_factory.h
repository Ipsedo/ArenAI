//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_OPENGL_FACTORY_H
#define ARENAI_OPENGL_FACTORY_H

#include <memory>
#include <string>

#include <arenai_view/factory.h>

#include "./drawables/gl_drawable_factory.h"
#include "./egl_render_context.h"
#include "./gl_hud_factory.h"

namespace arenai::view {

    // Headless OpenGL backend: owns an EGL context, produces offscreen renderers.
    // Also the base for the windowed backend (see src/glfw/glfw_opengl_factory.h).
    class OpenGlBackend : public virtual AbstractGraphicBackend {
    public:
        explicit OpenGlBackend(std::shared_ptr<EglRenderContext> context);

        std::shared_ptr<AbstractRenderContext> render_context() override;

        std::unique_ptr<AbstractOffscreenRenderer> make_offscreen_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) override;

        std::shared_ptr<AbstractDrawableFactory> drawable_factory() override;
        std::shared_ptr<AbstractHudFactory> hud_factory() override;

        std::string renderer_info() override;

        void release_thread() override;

    protected:
        // The backend owns the render context; renderers share it (window) or
        // derive from it (offscreen pbuffers). No downcast is ever needed.
        std::shared_ptr<EglRenderContext> context_;

    private:
        std::shared_ptr<GlDrawableFactory> drawable_factory_ =
            std::make_shared<GlDrawableFactory>();
        std::shared_ptr<GlHudFactory> hud_factory_ = std::make_shared<GlHudFactory>();
    };

    // Entry point for the OpenGL stack. make_headless_backend() is defined here;
    // make_windowed_backend() is GLFW-specific and lives in src/glfw so that this
    // OpenGL translation unit stays free of any windowing dependency.
    class OpenGlViewFactory final : public AbstractViewFactory {
    public:
        std::unique_ptr<AbstractGraphicBackend> make_headless_backend() override;

        std::unique_ptr<AbstractWindowedGraphicBackend> make_windowed_backend(
            int window_width, int window_height, const std::string &title) override;
    };

}// namespace arenai::view

#endif// ARENAI_OPENGL_FACTORY_H
