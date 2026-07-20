//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_BACKEND_H
#define ARENAI_VULKAN_BACKEND_H

#include <memory>
#include <string>

#include <arenai_view/backend.h>

#include "./core/device.h"
#include "./core/instance.h"
#include "./render_context.h"

namespace arenai::view {

    class VulkanDrawableFactory;
    class VulkanHudFactory;

    class VulkanBackend : public virtual AbstractGraphicBackend {
    public:
        // headless backend: no surface, prefers the integrated GPU so the
        // agent visions stay off the GPU driving the window
        VulkanBackend();

        std::shared_ptr<AbstractRenderContext> render_context() override;

        std::unique_ptr<AbstractOffscreenRenderer> make_offscreen_renderer(
            int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera) override;

        std::shared_ptr<AbstractDrawableFactory> drawable_factory() override;
        std::shared_ptr<AbstractHudFactory> hud_factory() override;

        std::string renderer_info() override;

        void release_thread() override;

    protected:
        // windowed subclass: instance carries the surface extensions, the
        // device is picked against the window surface
        VulkanBackend(std::shared_ptr<VulkanInstance> instance, const DeviceCriteria &criteria);

        const std::shared_ptr<VulkanRenderContext> &context() const;

    private:
        std::shared_ptr<VulkanRenderContext> context_;
        std::shared_ptr<VulkanDrawableFactory> drawable_factory_;
        std::shared_ptr<VulkanHudFactory> hud_factory_;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_BACKEND_H
