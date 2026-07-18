//
// Created by samuel on 17/07/2026.
//

#include "./glfw_vulkan_backend.h"

#include <utility>

#include "../vulkan/renderers/player_renderer.h"

namespace arenai::view {

    GlfwVulkanBackend::Bootstrap GlfwVulkanBackend::bootstrap(
        const int window_width, const int window_height, const std::string &title) {
        auto window = std::make_shared<GlfwVulkanWindow>(window_width, window_height, title);
        auto instance = std::make_shared<VulkanInstance>(window->required_instance_extensions());
        const VkSurfaceKHR surface = window->create_surface(instance->handle());
        return {std::move(window), std::move(instance), surface};
    }

    GlfwVulkanBackend::GlfwVulkanBackend(
        const int window_width, const int window_height, const std::string &title)
        : GlfwVulkanBackend(bootstrap(window_width, window_height, title)) {}

    GlfwVulkanBackend::GlfwVulkanBackend(Bootstrap bootstrap)
        : VulkanBackend(
            bootstrap.instance,
            DeviceCriteria{
                .prefer_integrated = false,
                .surface = bootstrap.surface,
                .device_env_var = "ARENAI_VK_DEVICE_WINDOW"}),
          window_(std::move(bootstrap.window)), surface_(bootstrap.surface) {
        frame_context_ =
            std::make_shared<WindowFrameContext>(context()->device(), surface_, [window = window_] {
                const auto [width, height] = window->framebuffer_size();
                return VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
            });
        rml_render_interface_ =
            std::make_unique<RmlVulkanRenderInterface>(context()->device(), frame_context_);
    }

    std::shared_ptr<AbstractWindow> GlfwVulkanBackend::get_window() { return window_; }

    std::unique_ptr<AbstractPlayerRenderer> GlfwVulkanBackend::make_player_renderer(
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera) {
        // query the size at creation time: the window may have been resized
        // since the backend was built (e.g. while the main menu was shown)
        const auto [width, height] = window_->framebuffer_size();
        return std::make_unique<VulkanPlayerRenderer>(
            context()->device(), frame_context_, width, height, light_pos, camera);
    }

    Rml::RenderInterface &GlfwVulkanBackend::ui_render_interface() {
        return *rml_render_interface_;
    }

    void GlfwVulkanBackend::begin_ui_frame(const int width, const int height) {
        if (!frame_context_->ensure_frame_begun()) return;

        frame_context_->begin_swapchain_pass(false, true);
        rml_render_interface_->begin_frame(width, height);
    }

    void GlfwVulkanBackend::begin_ui_overlay(const int width, const int height) {
        if (!frame_context_->ensure_frame_begun()) return;

        // no clear: the UI is composited over the frame already drawn
        frame_context_->begin_swapchain_pass(true, false);
        rml_render_interface_->begin_frame(width, height);
    }

    void GlfwVulkanBackend::end_ui_frame() {
        if (!frame_context_->frame_active()) return;

        rml_render_interface_->end_frame();
        frame_context_->end_swapchain_pass();
    }

    void GlfwVulkanBackend::present() { frame_context_->present(); }

    GlfwVulkanBackend::~GlfwVulkanBackend() {
        // the surface must outlive the swapchain, and the instance the surface
        rml_render_interface_.reset();
        frame_context_.reset();
        vkDestroySurfaceKHR(context()->instance()->handle(), surface_, nullptr);
    }

    /*
     * VulkanViewFactory: windowed backend construction (GLFW-specific).
     */

    std::unique_ptr<AbstractWindowedGraphicBackend> make_glfw_vulkan_backend(
        const int window_width, const int window_height, const std::string &title) {
        return std::make_unique<GlfwVulkanBackend>(window_width, window_height, title);
    }

}// namespace arenai::view
