//
// Created by samuel on 20/07/2026.
//

#include "./shadow_pass.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

namespace arenai::view {

    // maps light-space NDC to shadow-map coordinates: x/y from [-1, 1] to
    // [0, 1] with the y axis flipped (the depth pass rasterizes with a
    // negative-height viewport, so row 0 holds ndc.y = +1), z already in
    // [0, 1] with the zero-to-one ortho projection
    constexpr glm::mat4 SHADOW_BIAS_MATRIX(
        0.5f, 0.f, 0.f, 0.f, 0.f, -0.5f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.5f, 0.5f, 0.f, 1.f);

    ShadowPass::ShadowPass(
        std::shared_ptr<VulkanDevice> device, const glm::vec3 light_pos, const int nb_slots,
        const ShadowSettings &settings)
        : device_(std::move(device)), light_pos_(light_pos), settings_(settings), rings_(nb_slots) {
    }

    void ShadowPass::ensure_ready() {
        if (map_) return;
        const auto max_size = static_cast<int>(device_->properties().limits.maxImageDimension2D);
        map_ = std::make_unique<VulkanShadowMap>(device_, std::min(settings_.map_size, max_size));
    }

    glm::mat4 ShadowPass::light_view_projection(const glm::vec3 camera_pos) const {
        const glm::vec3 light_dir = glm::normalize(light_pos_);
        const auto up =
            std::abs(light_dir.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);

        const glm::mat4 light_view =
            glm::lookAt(light_dir * settings_.distance, glm::vec3(0.f), up);

        // center the ortho frustum on the camera, snapped to the shadow-map
        // texel grid to avoid shadow shimmering when the camera moves
        const auto center = glm::vec3(light_view * glm::vec4(camera_pos, 1.f));
        const float texel_size = 2.f * settings_.half_extent / static_cast<float>(map_->size());
        const float x = std::floor(center.x / texel_size) * texel_size;
        const float y = std::floor(center.y / texel_size) * texel_size;
        const float depth = -center.z;

        const glm::mat4 light_proj = glm::orthoRH_ZO(
            x - settings_.half_extent, x + settings_.half_extent, y - settings_.half_extent,
            y + settings_.half_extent, depth - settings_.depth_range,
            depth + settings_.depth_range);

        return light_proj * light_view;
    }

    glm::mat4 ShadowPass::biased(const glm::mat4 &light_vp_matrix) {
        return SHADOW_BIAS_MATRIX * light_vp_matrix;
    }

    void ShadowPass::begin_depth_pass(const VkCommandBuffer cmd) const {
        map_->begin_depth_pass(cmd);
    }

    void ShadowPass::end_depth_pass(const VkCommandBuffer cmd) const { map_->end_depth_pass(cmd); }

    bool ShadowPass::ensure_ring(const int slot, const size_t draw_count) {
        auto &ring = rings_[slot];

        const auto needed = static_cast<uint32_t>(std::max<size_t>(draw_count, 1));
        if (ring.capacity >= needed) return false;

        const uint32_t capacity = std::max(64u, std::bit_ceil(needed));
        ring.buffer = std::make_unique<HostVisibleBuffer>(
            device_, static_cast<size_t>(capacity) * settings_.ring_stride,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        ring.capacity = capacity;
        return true;
    }

    const HostVisibleBuffer &ShadowPass::ring(const int slot) const { return *rings_[slot].buffer; }

    uint32_t ShadowPass::stride() const { return settings_.ring_stride; }

    uint32_t ShadowPass::push_matrix(
        const int slot, const uint32_t index, const glm::mat4 &shadow_mvp_matrix) const {
        const uint32_t offset = index * settings_.ring_stride;
        auto *data = static_cast<uint8_t *>(rings_[slot].buffer->data());
        std::memcpy(data + offset, &shadow_mvp_matrix, sizeof(glm::mat4));
        return offset;
    }

    void ShadowPass::flush(const int slot) const {
        if (rings_[slot].buffer) rings_[slot].buffer->flush();
    }

    VkImageView ShadowPass::view() const { return map_->view(); }

    VkSampler ShadowPass::sampler() const { return map_->sampler(); }

    VkFormat ShadowPass::depth_format() const { return map_->format(); }

}// namespace arenai::view
