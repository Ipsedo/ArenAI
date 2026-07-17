//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_FACTORY_H
#define ARENAI_FACTORY_H

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "./camera.h"
#include "./drawable.h"
#include "./hud.h"
#include "./renderer.h"
#include "./window.h"

// Forward declaration only: RmlUi's render interface is the UI-side port this
// backend implements, but the library itself (like GL/GLFW) stays a PRIVATE
// dependency of arenai_view.
namespace Rml {
    class RenderInterface;
}

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
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera) = 0;

        // UI (RmlUi) support: the GL implementation of the render interface is
        // owned by the backend; callers install it into RmlCore and draw their
        // UI between begin_ui_frame()/begin_ui_overlay() and end_ui_frame(),
        // then present() the finished frame.
        virtual Rml::RenderInterface &ui_render_interface() = 0;
        // makes the window context current, clears the framebuffer and sets the
        // 2D UI render state for the given framebuffer size (UI-only frame)
        virtual void begin_ui_frame(int width, int height) = 0;
        // same, but without clearing: the UI is drawn on top of the frame
        // already rendered in the backbuffer (e.g. the pause menu over the game)
        virtual void begin_ui_overlay(int width, int height) = 0;
        // finishes the UI pass (the frame is not presented yet)
        virtual void end_ui_frame() = 0;
        // swaps the window buffers, displaying everything drawn so far
        virtual void present() = 0;
    };

    // The single symbol that "names" the OpenGL/GLFW stack on the application side.
    std::unique_ptr<AbstractGraphicBackend> make_opengl_backend();
    std::unique_ptr<AbstractWindowedGraphicBackend>
    make_glfw_backend(int window_width, int window_height, const std::string &title);

}// namespace arenai::view

#endif// ARENAI_FACTORY_H
