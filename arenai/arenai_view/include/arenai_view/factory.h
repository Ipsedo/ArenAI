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

    // A configured graphics backend: owns its render context and produces
    // offscreen renderers and drawables. Switching graphics engine = another
    // implementation. No GL/EGL/windowing type ever crosses this API.
    class AbstractGraphicBackend {
    public:
        virtual ~AbstractGraphicBackend() = default;

        virtual std::shared_ptr<AbstractRenderContext> render_context() = 0;

        virtual std::unique_ptr<AbstractOffscreenRenderer> make_offscreen_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) = 0;

        virtual std::shared_ptr<AbstractDrawableFactory> drawable_factory() = 0;
        virtual std::shared_ptr<AbstractHudFactory> hud_factory() = 0;

        // Human-readable description of the underlying GPU / driver.
        virtual std::string renderer_info() = 0;

        // Release any thread-local backend state before a worker thread exits.
        virtual void release_thread() = 0;
    };

    // A graphic backend that owns an on-screen window and can render to it.
    class AbstractWindowedGraphicBackend : public virtual AbstractGraphicBackend {
    public:
        virtual std::shared_ptr<AbstractWindow> get_window() = 0;

        virtual std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) = 0;
    };

    // Entry point for a graphics API: builds either a headless backend (offscreen
    // only, e.g. training) or a windowed backend (creates and owns the window).
    class AbstractViewFactory {
    public:
        virtual ~AbstractViewFactory() = default;

        virtual std::unique_ptr<AbstractGraphicBackend> make_headless_backend() = 0;

        virtual std::unique_ptr<AbstractWindowedGraphicBackend>
        make_windowed_backend(int window_width, int window_height, const std::string &title) = 0;
    };

    // The single symbol that "names" the OpenGL/GLFW stack on the application side.
    std::unique_ptr<AbstractViewFactory> make_opengl_view_factory();

}// namespace arenai::view

#endif// ARENAI_FACTORY_H
