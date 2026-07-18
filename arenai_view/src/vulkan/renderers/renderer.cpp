//
// Created by samuel on 17/07/2026.
//

#include "./renderer.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "../drawables/shadow_drawable.h"
#include "../drawables/vulkan_drawable.h"

using namespace arenai;

namespace arenai::view {

    // maps light-space NDC to shadow-map coordinates: x/y from [-1, 1] to
    // [0, 1] with the y axis flipped (the depth pass rasterizes with a
    // negative-height viewport, so row 0 holds ndc.y = +1), z already in
    // [0, 1] with the zero-to-one ortho projection
    constexpr glm::mat4 SHADOW_BIAS_MATRIX(
        0.5f, 0.f, 0.f, 0.f, 0.f, -0.5f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.5f, 0.5f, 0.f, 1.f);

    // matches the horizon of the "cubemap/1" sky, keeps distant geometry
    // blending into it (same constant as the GL VulkanDiffuse drawable, lifted to
    // the frame globals UBO)
    constexpr glm::vec3 FOG_COLOR(0.53f, 0.57f, 0.65f);

    namespace {
        struct FrameGlobals {
            glm::vec4 light_pos;
            glm::vec4 world_up;
            glm::vec4 fog_color;
        };
    }// namespace

    VulkanRenderer::VulkanRenderer(
        std::shared_ptr<VulkanDevice> device, const glm::vec3 light_pos,
        std::shared_ptr<AbstractCamera> camera, const bool with_shadows)
        : device_(std::move(device)), light_pos_(light_pos), with_shadows_(with_shadows),
          camera_(std::move(camera)), upload_pool_(device_->make_command_pool()),
          descriptors_(std::make_unique<DescriptorAllocator>(device_)),
          set0_plain_layout_(VK_NULL_HANDLE), set0_shadow_layout_(VK_NULL_HANDLE) {
        set0_plain_layout_ = DescriptorLayoutBuilder()
                                 .add_binding(
                                     0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                                 .build(device_->handle());
        set0_shadow_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(
                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                .add_binding(
                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT)
                .add_binding(
                    2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(device_->handle());
    }

    void VulkanRenderer::add_drawable(
        const std::string &name, std::unique_ptr<AbstractDrawable> drawable) {
        if (auto *vulkan_drawable = dynamic_cast<VulkanDrawable *>(drawable.get()))
            vulkan_drawable->attach(this);
        drawables_.insert({name, std::move(drawable)});
    }

    void VulkanRenderer::remove_drawable(const std::string &name) {
        const auto entry = drawables_.find(name);
        if (entry == drawables_.end()) return;
        retired_.emplace_back(frame_counter_, std::move(entry->second));
        drawables_.erase(entry);
    }

    void VulkanRenderer::drain_retired(const bool everything) {
        std::erase_if(retired_, [&](const auto &entry) {
            return everything || frame_counter_ - entry.first >= FRAME_SLOTS;
        });
    }

    glm::mat4 VulkanRenderer::light_view_projection() const {
        const glm::vec3 light_dir = glm::normalize(light_pos_);
        const auto up =
            std::abs(light_dir.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);

        const glm::mat4 light_view = glm::lookAt(light_dir * SHADOW_DISTANCE, glm::vec3(0.f), up);

        // center the ortho frustum on the camera, snapped to the shadow-map
        // texel grid to avoid shadow shimmering when the camera moves
        const auto center = glm::vec3(light_view * glm::vec4(camera_->pos(), 1.f));
        const float texel_size = 2.f * SHADOW_HALF_EXTENT / static_cast<float>(shadow_map_->size());
        const float x = std::floor(center.x / texel_size) * texel_size;
        const float y = std::floor(center.y / texel_size) * texel_size;
        const float depth = -center.z;

        const glm::mat4 light_proj = glm::orthoRH_ZO(
            x - SHADOW_HALF_EXTENT, x + SHADOW_HALF_EXTENT, y - SHADOW_HALF_EXTENT,
            y + SHADOW_HALF_EXTENT, depth - SHADOW_DEPTH_RANGE, depth + SHADOW_DEPTH_RANGE);

        return light_proj * light_view;
    }

    void VulkanRenderer::ensure_slot_resources(const int slot, const size_t draw_count) {
        auto &resources = slots_[slot];

        if (!resources.globals) {
            resources.globals = std::make_unique<HostVisibleBuffer>(
                device_, sizeof(FrameGlobals), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            resources.set0_plain = descriptors_->allocate(set0_plain_layout_);
            write_buffer_descriptor(
                device_->handle(), resources.set0_plain, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                resources.globals->handle(), 0, sizeof(FrameGlobals));
        }

        if (!with_shadows_) return;

        const auto needed = static_cast<uint32_t>(std::max<size_t>(draw_count, 1));
        if (resources.shadow_ring_capacity < needed) {
            const uint32_t capacity = std::max(64u, std::bit_ceil(needed));
            resources.shadow_ring = std::make_unique<HostVisibleBuffer>(
                device_, static_cast<size_t>(capacity) * SHADOW_RING_STRIDE,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            resources.shadow_ring_capacity = capacity;

            if (resources.set0_shadow == VK_NULL_HANDLE) {
                resources.set0_shadow = descriptors_->allocate(set0_shadow_layout_);
                write_buffer_descriptor(
                    device_->handle(), resources.set0_shadow, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    resources.globals->handle(), 0, sizeof(FrameGlobals));
                write_image_descriptor(
                    device_->handle(), resources.set0_shadow, 2, shadow_map_->sampler(),
                    shadow_map_->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            write_buffer_descriptor(
                device_->handle(), resources.set0_shadow, 1,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, resources.shadow_ring->handle(), 0,
                sizeof(glm::mat4));
        }
    }

    void
    VulkanRenderer::draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        const auto [cmd, slot] = on_begin_frame();
        // minimized window: the whole frame is skipped
        if (cmd == VK_NULL_HANDLE) return;
        frame_.cmd = cmd;
        frame_.slot = slot;
        frame_counter_++;
        drain_retired(false);

        if (with_shadows_ && !shadow_map_) {
            const auto max_size =
                static_cast<int>(device_->properties().limits.maxImageDimension2D);
            shadow_map_ =
                std::make_unique<VulkanShadowMap>(device_, std::min(SHADOW_MAP_SIZE, max_size));
        }

        ensure_slot_resources(slot, model_matrices.size());
        frame_.set0_plain = slots_[slot].set0_plain;
        frame_.set0_shadow = slots_[slot].set0_shadow;

        glm::mat4 light_vp_matrix(1.f);
        bool shadow_pass_done = false;

        if (with_shadows_) {
            light_vp_matrix = light_view_projection();

            shadow_map_->begin_depth_pass(cmd);
            for (const auto &[name, m_matrix]: model_matrices)
                if (auto *shadow_drawable =
                        dynamic_cast<VulkanShadowDrawable *>(drawables_.at(name).get()))
                    shadow_drawable->draw_depth(light_vp_matrix * m_matrix);
            shadow_map_->end_depth_pass(cmd);

            shadow_pass_done = true;
        }

        // on_draw
        const auto camera_pos = camera_->pos();
        const glm::mat4 view_matrix = glm::lookAt(camera_pos, camera_->look(), camera_->up());

        // zero-to-one depth projection (Vulkan clip space); the y flip is
        // handled by the negative-height viewport, not by the matrix
        const glm::mat4 proj_matrix = glm::perspectiveRH_ZO(
            static_cast<float>(M_PI) / 4.f,
            static_cast<float>(get_width()) / static_cast<float>(get_height()), 1.f,
            2000.f * std::sqrt(3.f));

        const glm::mat4 biased_light_vp_matrix = SHADOW_BIAS_MATRIX * light_vp_matrix;

        // world up axis in view space (xyz) + camera world height (w): enough
        // for the shadow shader to do hemisphere ambient and height fog
        // without a full inverse view matrix
        const glm::vec4 world_up(glm::mat3(view_matrix) * glm::vec3(0.f, 1.f, 0.f), camera_pos.y);

        const FrameGlobals globals{glm::vec4(light_pos_, 0.f), world_up, glm::vec4(FOG_COLOR, 0.f)};
        std::memcpy(slots_[slot].globals->data(), &globals, sizeof(FrameGlobals));

        on_begin_scene_pass();

        uint32_t shadow_index = 0;
        for (const auto &[name, m_matrix]: model_matrices) {
            auto mv_matrix = view_matrix * m_matrix;
            const auto mvp_matrix = proj_matrix * mv_matrix;

            auto *shadow_drawable =
                shadow_pass_done ? dynamic_cast<VulkanShadowDrawable *>(drawables_.at(name).get())
                                 : nullptr;

            if (shadow_drawable) {
                const glm::mat4 shadow_mvp_matrix = biased_light_vp_matrix * m_matrix;
                auto *ring = static_cast<uint8_t *>(slots_[slot].shadow_ring->data());
                std::memcpy(
                    ring + shadow_index * SHADOW_RING_STRIDE, &shadow_mvp_matrix,
                    sizeof(glm::mat4));
                frame_.shadow_dynamic_offset = shadow_index * SHADOW_RING_STRIDE;
                shadow_index++;

                shadow_drawable->draw_with_shadow(
                    mvp_matrix, mv_matrix, light_pos_, camera_pos, world_up);
            } else drawables_.at(name)->draw(mvp_matrix, mv_matrix, light_pos_, camera_pos);
        }

        slots_[slot].globals->flush();
        if (slots_[slot].shadow_ring) slots_[slot].shadow_ring->flush();

        on_end_frame(view_matrix, proj_matrix);
    }

    void VulkanRenderer::make_current() const {}

    void VulkanRenderer::release_current() const {}

    const VulkanRenderer::SceneFrame &VulkanRenderer::scene_frame() const { return frame_; }

    const std::shared_ptr<VulkanDevice> &VulkanRenderer::device() const { return device_; }

    VkCommandPool VulkanRenderer::upload_pool() const { return upload_pool_; }

    DescriptorAllocator &VulkanRenderer::descriptors() { return *descriptors_; }

    VkDescriptorSetLayout VulkanRenderer::set0_plain_layout() const { return set0_plain_layout_; }

    VkDescriptorSetLayout VulkanRenderer::set0_shadow_layout() const { return set0_shadow_layout_; }

    VkFormat VulkanRenderer::shadow_depth_format() const { return shadow_map_->format(); }

    const glm::vec3 &VulkanRenderer::light_position() const { return light_pos_; }

    const std::shared_ptr<AbstractCamera> &VulkanRenderer::camera() const { return camera_; }

    VulkanRenderer::~VulkanRenderer() {
        // the subclass destructor has already waited its frame fences
        drain_retired(true);
        drawables_.clear();
        shadow_map_.reset();
        for (auto &resources: slots_) {
            resources.globals.reset();
            resources.shadow_ring.reset();
        }
        descriptors_.reset();
        vkDestroyDescriptorSetLayout(device_->handle(), set0_shadow_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_->handle(), set0_plain_layout_, nullptr);
        vkDestroyCommandPool(device_->handle(), upload_pool_, nullptr);
    }

}// namespace arenai::view
