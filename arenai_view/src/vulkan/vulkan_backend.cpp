//
// Created by samuel on 17/07/2026.
//

#include "./vulkan_backend.h"

#include <utility>

#include "./scene/drawables/drawable_factory.h"
#include "./scene/drawables/hud_factory.h"
#include "./scene/renderers/offscreen_renderer.h"

namespace arenai::view {

    /*
     * VulkanBackend (headless)
     */

    VulkanBackend::VulkanBackend()
        : VulkanBackend(
            std::make_shared<VulkanInstance>(), DeviceCriteria{
                                                    .prefer_integrated = true,
                                                    .surface = VK_NULL_HANDLE,
                                                    .device_env_var = "ARENAI_VK_DEVICE"}) {}

    VulkanBackend::VulkanBackend(
        std::shared_ptr<VulkanInstance> instance, const DeviceCriteria &criteria)
        : context_(std::make_shared<VulkanRenderContext>(
            instance, std::make_shared<VulkanDevice>(instance, criteria))),
          drawable_factory_(std::make_shared<VulkanDrawableFactory>()),
          hud_factory_(std::make_shared<VulkanHudFactory>()) {}

    std::shared_ptr<AbstractRenderContext> VulkanBackend::render_context() { return context_; }

    std::unique_ptr<AbstractOffscreenRenderer> VulkanBackend::make_offscreen_renderer(
        const int width, const int height, const glm::vec3 light_pos,
        const std::shared_ptr<AbstractCamera> &camera) {
        return std::make_unique<VulkanOffscreenRenderer>(
            context_->device(), width, height, light_pos, camera);
    }

    std::shared_ptr<AbstractDrawableFactory> VulkanBackend::drawable_factory() {
        return drawable_factory_;
    }

    std::shared_ptr<AbstractHudFactory> VulkanBackend::hud_factory() { return hud_factory_; }

    std::string VulkanBackend::renderer_info() { return context_->device()->renderer_info(); }

    void VulkanBackend::release_thread() {
        // no thread-bound state in Vulkan
    }

    const std::shared_ptr<VulkanRenderContext> &VulkanBackend::context() const { return context_; }

    /*
     * VulkanViewFactory (headless part; windowed part in src/glfw)
     */

    std::unique_ptr<AbstractGraphicBackend> make_vulkan_backend() {
        return std::make_unique<VulkanBackend>();
    }

}// namespace arenai::view
