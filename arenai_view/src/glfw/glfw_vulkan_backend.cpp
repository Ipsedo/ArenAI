//
// Created by samuel on 17/07/2026.
//

#include "./glfw_vulkan_backend.h"

#include <utility>

#include "../vulkan/scene/renderers/player_renderer.h"

namespace arenai::view {

    GlfwVulkanBackend::Bootstrap GlfwVulkanBackend::bootstrap(
        const int window_width, const int window_height, const std::string &title,
        const std::string &gpu_name) {
        auto window = std::make_shared<GlfwVulkanWindow>(window_width, window_height, title);
        auto instance = std::make_shared<VulkanInstance>(window->required_instance_extensions());
        const VkSurfaceKHR &surface = window->create_surface(instance->handle());
        return {
            .window = std::move(window),
            .instance = std::move(instance),
            .surface = surface,
            .gpu_name = gpu_name};
    }

    GlfwVulkanBackend::GlfwVulkanBackend(
        const int window_width, const int window_height, const std::string &title,
        const std::string &gpu_name)
        : GlfwVulkanBackend(bootstrap(window_width, window_height, title, gpu_name)) {}

    GlfwVulkanBackend::GlfwVulkanBackend(Bootstrap bootstrap)
        : VulkanBackend(
            bootstrap.instance,
            DeviceCriteria{
                .prefer_integrated = false,
                .surface = bootstrap.surface,
                .device_env_var = "ARENAI_VK_DEVICE_WINDOW",
                .preferred_device = bootstrap.gpu_name}),
          window_(std::move(bootstrap.window)), surface_(bootstrap.surface) {
        frame_context_ =
            std::make_shared<WindowFrameContext>(context()->device(), surface_, [window = window_] {
                const auto [width, height] = window->framebuffer_size();
                return VkExtent2D{
                    .width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height)};
            });
        rml_render_interface_ =
            std::make_unique<RmlVulkanRenderInterface>(context()->device(), frame_context_);
    }

    std::shared_ptr<AbstractWindow> GlfwVulkanBackend::get_window() { return window_; }

    std::unique_ptr<AbstractPlayerRenderer> GlfwVulkanBackend::make_player_renderer(
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
        const PlayerRendererSettings &settings) {

        const auto [width, height] = window_->framebuffer_size();
        return std::make_unique<VulkanPlayerRenderer>(
            context()->device(), frame_context_, width, height, light_pos, camera, settings.shadows,
            settings.shadow_map_size, settings.msaa_samples);
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
        rml_render_interface_.reset();
        frame_context_.reset();
        vkDestroySurfaceKHR(context()->instance()->handle(), surface_, nullptr);
    }

    /*
     * VulkanViewFactory: windowed backend construction (GLFW-specific).
     */

    std::unique_ptr<AbstractWindowedGraphicBackend> make_glfw_vulkan_backend(
        const int window_width, const int window_height, const std::string &title,
        const std::string &gpu_name) {
        return std::make_unique<GlfwVulkanBackend>(window_width, window_height, title, gpu_name);
    }

}// namespace arenai::view
