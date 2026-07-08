//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_FACTORY_H
#define ARENAI_FACTORY_H

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "./camera.h"
#include "./drawable_factory.h"
#include "./hud_factory.h"
#include "./render_context.h"
#include "./renderer.h"
#include "./window.h"

namespace arenai::view {

    class AbstractGraphicBackend {
    public:
        virtual ~AbstractGraphicBackend() = default;

        virtual std::shared_ptr<AbstractRenderContext> render_context() = 0;

        virtual std::unique_ptr<AbstractOffscreenRenderer> make_offscreen_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) = 0;

        virtual std::shared_ptr<AbstractDrawableFactory> drawable_factory() = 0;
        virtual std::shared_ptr<AbstractHudFactory> hud_factory() = 0;

        virtual std::string renderer_info() = 0;

        virtual void release_thread() = 0;
    };

    class AbstractWindowedGraphicBackend : public virtual AbstractGraphicBackend {
    public:
        virtual std::shared_ptr<AbstractWindow> get_window() = 0;

        virtual std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) = 0;
    };

    // The single symbol that "names" the OpenGL/GLFW stack on the application side.
    std::unique_ptr<AbstractGraphicBackend> make_opengl_backend();
    std::unique_ptr<AbstractWindowedGraphicBackend>
    make_glfw_backend(int window_width, int window_height, const std::string &title);

}// namespace arenai::view

#endif// ARENAI_FACTORY_H
