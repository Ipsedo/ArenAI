//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_OPENGL_FACTORY_H
#define ARENAI_OPENGL_FACTORY_H

#include <memory>
#include <string>

#include <arenai_view/backend.h>

#include "./drawables/gl_drawable_factory.h"
#include "./drawables/gl_hud_factory.h"
#include "./egl_render_context.h"

namespace arenai::view {

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
        std::shared_ptr<EglRenderContext> context_;

    private:
        std::shared_ptr<GlDrawableFactory> drawable_factory_ =
            std::make_shared<GlDrawableFactory>();
        std::shared_ptr<GlHudFactory> hud_factory_ = std::make_shared<GlHudFactory>();
    };

}// namespace arenai::view

#endif// ARENAI_OPENGL_FACTORY_H
