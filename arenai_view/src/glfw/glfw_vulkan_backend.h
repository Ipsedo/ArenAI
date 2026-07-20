//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_GLFW_VULKAN_BACKEND_H
#define ARENAI_GLFW_VULKAN_BACKEND_H

#include <memory>
#include <string>

#include <arenai_view/backend.h>

#include "../vulkan/present/window_frame.h"
#include "../vulkan/ui/rml_render_interface.h"
#include "../vulkan/vulkan_backend.h"
#include "./glfw_vulkan_window.h"

namespace arenai::view {

    class GlfwVulkanBackend final : public VulkanBackend, public AbstractWindowedGraphicBackend {
    public:
        GlfwVulkanBackend(int window_width, int window_height, const std::string &title);

        std::shared_ptr<AbstractWindow> get_window() override;

        std::unique_ptr<AbstractPlayerRenderer> make_player_renderer(
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera) override;

        Rml::RenderInterface &ui_render_interface() override;
        void begin_ui_frame(int width, int height) override;
        void begin_ui_overlay(int width, int height) override;
        void end_ui_frame() override;
        void present() override;

        ~GlfwVulkanBackend() override;

    private:
        // window + instance + surface exist before the device is picked
        // against the surface: built ahead of the base-class construction
        struct Bootstrap {
            std::shared_ptr<GlfwVulkanWindow> window;
            std::shared_ptr<VulkanInstance> instance;
            VkSurfaceKHR surface;
        };
        static Bootstrap bootstrap(int window_width, int window_height, const std::string &title);
        explicit GlfwVulkanBackend(Bootstrap bootstrap);

        std::shared_ptr<GlfwVulkanWindow> window_;
        VkSurfaceKHR surface_;

        std::shared_ptr<WindowFrameContext> frame_context_;
        std::unique_ptr<RmlVulkanRenderInterface> rml_render_interface_;
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_VULKAN_BACKEND_H
