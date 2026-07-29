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

    // Depth-only shadow map with a hardware-PCF compare sampler (LINEAR +
    // LESS_OR_EQUAL, clamp to edge — mirrors the GL sampler setup). The
    // slope-scaled depth bias lives in the depth pipelines (2, 4), not here.
    class VulkanShadowMap {
    public:
        VulkanShadowMap(const std::shared_ptr<VulkanDevice> &device, int size);

        VulkanShadowMap(const VulkanShadowMap &) = delete;
        VulkanShadowMap &operator=(const VulkanShadowMap &) = delete;

        // barrier + begin depth-only rendering + viewport/scissor; the
        // negative-height viewport keeps the scene winding, the y-flip is
        // compensated in the shadow bias matrix
        void begin_depth_pass(VkCommandBuffer cmd) const;
        // end rendering + barrier to fragment-shader sampling
        void end_depth_pass(VkCommandBuffer cmd) const;

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
