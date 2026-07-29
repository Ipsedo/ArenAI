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

    // Every dial of the directional shadow pass; the defaults are the values
    // tuned for the arena.
    struct ShadowSettings {
        // requested shadow-map resolution, clamped to the device 2D limit
        int map_size = 16384;
        // ortho frustum half extent, centered on the camera (the arena is far
        // too large to be covered by a single shadow map at a usable resolution)
        float half_extent = 500.f;
        float distance = 1000.f;
        // must cover the light-space depth spread of the whole frustum: with a
        // ~47° light elevation, ground at the frustum corners reaches ~±500
        float depth_range = 900.f;
        // minUniformBufferOffsetAlignment-safe stride of the matrix ring
        uint32_t ring_stride = 256;
    };

    // The directional shadow pass of a scene renderer: owns the shadow map
    // (created lazily on first frame), the light-space matrices (ortho
    // frustum snapped to the shadow-map texel grid) and the per-slot ring of
    // per-draw shadow matrices. The renderer keeps ownership of the drawables
    // and of the set-0 descriptors; this class only feeds them.
    class ShadowPass {
    public:
        ShadowPass(
            std::shared_ptr<VulkanDevice> device, glm::vec3 light_pos, int nb_slots,
            const ShadowSettings &settings = {});

        ShadowPass(const ShadowPass &) = delete;
        ShadowPass &operator=(const ShadowPass &) = delete;

        // creates the shadow map on first call; no-op afterwards
        void ensure_ready();

        glm::mat4 light_view_projection(glm::vec3 camera_pos) const;
        // maps light-space NDC to shadow-map coordinates (see SHADOW_BIAS_MATRIX)
        static glm::mat4 biased(const glm::mat4 &light_vp_matrix);

        void begin_depth_pass(VkCommandBuffer cmd) const;
        void end_depth_pass(VkCommandBuffer cmd) const;

        // grows the slot's matrix ring to hold draw_count entries; returns
        // true when the buffer was (re)allocated and the descriptor bound to
        // it must be (re)written
        bool ensure_ring(int slot, size_t draw_count);
        const HostVisibleBuffer &ring(int slot) const;
        uint32_t stride() const;
        // writes the matrix into the slot's ring, returns its dynamic offset
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
