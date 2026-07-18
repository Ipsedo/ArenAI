//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_RENDERER_H
#define ARENAI_VK_RENDERER_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_view/camera.h>
#include <arenai_view/renderer.h>

#include "../buffer.h"
#include "../descriptors.h"
#include "../device.h"
#include "../shadow_map.h"

namespace arenai::view {

    // Base scene renderer: same orchestration as the GL one (shadow depth
    // pass, then the scene pass through the drawables map), rebuilt on the
    // frame model of Vulkan. Subclasses own the frame slots (command buffer,
    // fence, targets) and expose them through the on_* hooks; the base owns
    // everything the drawables consume: the per-slot set-0 descriptors (frame
    // globals UBO, dynamic shadow-matrix ring, shadow-map sampler), the
    // thread-confined upload pool and the descriptor allocator.
    class VulkanRenderer : public virtual AbstractRenderer {
    public:
        static constexpr int FRAME_SLOTS = 2;

        VulkanRenderer(
            std::shared_ptr<VulkanDevice> device, glm::vec3 light_pos,
            std::shared_ptr<AbstractCamera> camera, bool with_shadows);
        ~VulkanRenderer() override;

        void
        add_drawable(const std::string &name, std::unique_ptr<AbstractDrawable> drawable) override;
        void remove_drawable(const std::string &name) override;

        void draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        // Vulkan has no thread-bound context: both are no-ops
        void make_current() const override;
        void release_current() const override;

        /*
         * internal API consumed by the drawables
         */

        struct SceneFrame {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            int slot = 0;
            VkDescriptorSet set0_plain = VK_NULL_HANDLE;
            VkDescriptorSet set0_shadow = VK_NULL_HANDLE;
            // 256-aligned offset of the current draw's shadow matrix in the ring
            uint32_t shadow_dynamic_offset = 0;
        };

        const SceneFrame &scene_frame() const;
        const std::shared_ptr<VulkanDevice> &device() const;
        // thread-confined pool for the drawables' one-shot uploads
        VkCommandPool upload_pool() const;
        DescriptorAllocator &descriptors();
        VkDescriptorSetLayout set0_plain_layout() const;
        VkDescriptorSetLayout set0_shadow_layout() const;
        VkFormat shadow_depth_format() const;

        // scene-pass attachment setup, for the drawables' lazy pipelines
        virtual VkFormat scene_color_format() const = 0;
        virtual VkFormat scene_depth_format() const = 0;
        virtual VkSampleCountFlagBits scene_samples() const = 0;

    protected:
        // waits/reuses a frame slot and returns its begun command buffer
        virtual std::pair<VkCommandBuffer, int> on_begin_frame() = 0;
        // begins the scene rendering pass (attachments, viewport, scissor)
        virtual void on_begin_scene_pass() = 0;
        // ends the scene pass; the subclass decides what follows (readback
        // copy + submit offscreen, post-processing + HUD for the player)
        virtual void on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) = 0;

        const glm::vec3 &light_position() const;
        const std::shared_ptr<AbstractCamera> &camera() const;
        // drops every drawable retired at least FRAME_SLOTS frames ago; the
        // subclass destructor calls it after waiting its fences
        void drain_retired(bool everything);

    private:
        static constexpr int SHADOW_MAP_SIZE = 16384;
        // ortho frustum half extent, centered on the camera (the arena is far
        // too large to be covered by a single shadow map at a usable resolution)
        static constexpr float SHADOW_HALF_EXTENT = 500.f;
        static constexpr float SHADOW_DISTANCE = 1000.f;
        // must cover the light-space depth spread of the whole frustum: with a
        // ~47° light elevation, ground at the frustum corners reaches ~±500
        static constexpr float SHADOW_DEPTH_RANGE = 900.f;

        static constexpr uint32_t SHADOW_RING_STRIDE = 256;

        struct SlotResources {
            std::unique_ptr<HostVisibleBuffer> globals;
            std::unique_ptr<HostVisibleBuffer> shadow_ring;
            uint32_t shadow_ring_capacity = 0;
            VkDescriptorSet set0_plain = VK_NULL_HANDLE;
            VkDescriptorSet set0_shadow = VK_NULL_HANDLE;
        };

        glm::mat4 light_view_projection() const;
        void ensure_slot_resources(int slot, size_t draw_count);

        std::shared_ptr<VulkanDevice> device_;

        glm::vec3 light_pos_;
        bool with_shadows_;
        std::unique_ptr<VulkanShadowMap> shadow_map_;

        std::map<std::string, std::unique_ptr<AbstractDrawable>> drawables_;
        // deferred destruction: a removed drawable may still be referenced by
        // an in-flight frame; it is destroyed FRAME_SLOTS frames later
        std::vector<std::pair<uint64_t, std::unique_ptr<AbstractDrawable>>> retired_;
        uint64_t frame_counter_ = 0;

        std::shared_ptr<AbstractCamera> camera_;

        VkCommandPool upload_pool_;
        std::unique_ptr<DescriptorAllocator> descriptors_;
        VkDescriptorSetLayout set0_plain_layout_;
        VkDescriptorSetLayout set0_shadow_layout_;

        SlotResources slots_[FRAME_SLOTS];
        SceneFrame frame_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_RENDERER_H
