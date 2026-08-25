//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_RENDER_CONTEXT_H
#define ARENAI_VK_RENDER_CONTEXT_H

#include <memory>

#include <arenai_view/renderer.h>

#include "./core/device.h"
#include "./core/instance.h"

namespace arenai::view {

    class VulkanRenderContext final : public AbstractRenderContext {
    public:
        VulkanRenderContext(
            std::shared_ptr<VulkanInstance> instance, std::shared_ptr<VulkanDevice> device);

        void make_current() override;
        void release_current() override;

        const std::shared_ptr<VulkanDevice> &device() const;
        const std::shared_ptr<VulkanInstance> &instance() const;

    private:
        std::shared_ptr<VulkanInstance> instance_;
        std::shared_ptr<VulkanDevice> device_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_RENDER_CONTEXT_H
