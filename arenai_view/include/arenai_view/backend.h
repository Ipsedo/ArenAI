//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_FACTORY_H
#define ARENAI_FACTORY_H

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "./camera.h"
#include "./drawable.h"
#include "./renderer.h"
#include "./window.h"

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

        virtual std::string renderer_info() = 0;

        virtual void release_thread() = 0;
    };

    enum class UiPass { Clear, Overlay };

    struct PlayerRendererSettings {
        bool shadows = true;
        int shadow_map_size = 16384;
        int msaa_samples = 4;
    };

    class AbstractWindowedGraphicBackend : public virtual AbstractGraphicBackend {
    public:
        virtual std::shared_ptr<AbstractWindow> get_window() = 0;

        virtual std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
            const PlayerRendererSettings &settings) = 0;

        virtual Rml::RenderInterface &ui_render_interface() = 0;

        // a UI pass on the swapchain image: Clear starts from a cleared
        // image (the UI is the whole frame), Overlay draws over the frame
        // already rendered
        virtual void begin_ui_pass(UiPass pass, int width, int height) = 0;
        virtual void end_ui_pass() = 0;

        virtual void present() = 0;
    };

    std::unique_ptr<AbstractGraphicBackend> make_vulkan_backend(const std::string &gpu_name = "");
    std::unique_ptr<AbstractWindowedGraphicBackend> make_glfw_vulkan_backend(
        int window_width, int window_height, const std::string &title,
        const std::string &gpu_name = "");

    std::vector<std::string> list_vulkan_gpus();

}// namespace arenai::view

#endif// ARENAI_FACTORY_H
