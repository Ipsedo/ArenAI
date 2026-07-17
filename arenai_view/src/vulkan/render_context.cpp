//
// Created by samuel on 17/07/2026.
//

#include "./render_context.h"

#include <utility>

namespace arenai::view {

    VulkanRenderContext::VulkanRenderContext(
        std::shared_ptr<VulkanInstance> instance, std::shared_ptr<VulkanDevice> device)
        : instance_(std::move(instance)), device_(std::move(device)) {}

    void VulkanRenderContext::make_current() {}

    void VulkanRenderContext::release_current() {}

    const std::shared_ptr<VulkanDevice> &VulkanRenderContext::device() const { return device_; }

    const std::shared_ptr<VulkanInstance> &VulkanRenderContext::instance() const {
        return instance_;
    }

}// namespace arenai::view
