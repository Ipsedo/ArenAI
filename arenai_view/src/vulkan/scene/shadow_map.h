//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SHADOW_MAP_H
#define ARENAI_VK_SHADOW_MAP_H

#include <memory>

#include "../core/device.h"
#include "../core/render_target.h"
#include "../core/vk.h"

namespace arenai::view {

    class VulkanShadowMap {
    public:
        VulkanShadowMap(const std::shared_ptr<VulkanDevice> &device, int size);

        VulkanShadowMap(const VulkanShadowMap &) = delete;
        VulkanShadowMap &operator=(const VulkanShadowMap &) = delete;

        void begin_depth_pass(const VkCommandBuffer &cmd) const;

        void end_depth_pass(const VkCommandBuffer &cmd) const;

        VkImageView view() const;
        VkSampler sampler() const;
        VkFormat format() const;

        int size() const;

        ~VulkanShadowMap();

    private:
        std::shared_ptr<VulkanDevice> device_;
        std::unique_ptr<Target> depth_;
        VkSampler sampler_;
        int size_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SHADOW_MAP_H
