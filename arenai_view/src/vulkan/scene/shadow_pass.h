//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_SHADOW_PASS_H
#define ARENAI_VK_SHADOW_PASS_H

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "../core/buffer.h"
#include "../core/device.h"
#include "../core/vk.h"
#include "./shadow_map.h"

namespace arenai::view {

    struct ShadowSettings {
        int map_size = 16384;
        float half_extent = 500.f;
        float distance = 1000.f;
        float depth_range = 900.f;
        uint32_t ring_stride = 256;
    };

    class ShadowPass {
    public:
        ShadowPass(
            std::shared_ptr<VulkanDevice> device, glm::vec3 light_pos, int nb_slots,
            const ShadowSettings &settings = {});

        ShadowPass(const ShadowPass &) = delete;
        ShadowPass &operator=(const ShadowPass &) = delete;

        void ensure_ready();

        glm::mat4 light_view_projection(glm::vec3 camera_pos) const;

        static glm::mat4 biased(const glm::mat4 &light_vp_matrix);

        void begin_depth_pass(const VkCommandBuffer &cmd) const;
        void end_depth_pass(const VkCommandBuffer &cmd) const;

        bool ensure_ring(int slot, size_t draw_count);
        const HostVisibleBuffer &ring(int slot) const;
        uint32_t stride() const;

        uint32_t push_matrix(int slot, uint32_t index, const glm::mat4 &shadow_mvp_matrix) const;
        void flush(int slot) const;

        VkImageView view() const;
        VkSampler sampler() const;
        VkFormat depth_format() const;

    private:
        struct SlotRing {
            std::unique_ptr<HostVisibleBuffer> buffer;
            uint32_t capacity = 0;
        };

        std::shared_ptr<VulkanDevice> device_;
        glm::vec3 light_pos_;
        ShadowSettings settings_;

        std::unique_ptr<VulkanShadowMap> map_;
        std::vector<SlotRing> rings_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SHADOW_PASS_H
