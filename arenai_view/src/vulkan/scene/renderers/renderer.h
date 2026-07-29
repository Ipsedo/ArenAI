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

#include "../../core/buffer.h"
#include "../../core/descriptors.h"
#include "../../core/device.h"
#include "../../core/retire_queue.h"
#include "../drawables/drawable_context.h"
#include "../shadow_pass.h"

namespace arenai::view {

    class VulkanShadowDrawable;

    // Base scene renderer: same orchestration as the GL one (shadow depth
    // pass, then the scene pass through the drawables map), rebuilt on the
    // frame model of Vulkan. Subclasses own the frame slots (command buffer,
    // fence, targets) and expose them through the on_* hooks; the base owns
    // the drawable registry, the per-slot set-0 descriptors (frame globals
    // UBO, dynamic shadow-matrix ring, shadow-map sampler), the
    // thread-confined upload pool and the descriptor allocator. Everything
    // the drawables consume goes through the DrawableContext port.
    class VulkanRenderer : public virtual AbstractRenderer, public DrawableContext {
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
         * DrawableContext port
         */

        const SceneFrame &scene_frame() const override;
        const std::shared_ptr<VulkanDevice> &device() const override;
        VkCommandPool upload_pool() const override;
        DescriptorAllocator &descriptors() override;
        VkDescriptorSetLayout set0_plain_layout() const override;
        VkDescriptorSetLayout set0_shadow_layout() const override;
        VkFormat shadow_depth_format() const override;
        // scene_color_format/scene_depth_format/scene_samples stay pure: the
        // subclasses own the scene attachments

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

    private:
        // a drawable and its shadow capability, resolved once at add time
        struct DrawableEntry {
            std::unique_ptr<AbstractDrawable> drawable;
            VulkanShadowDrawable *shadow = nullptr;
        };

        struct SlotResources {
            std::unique_ptr<HostVisibleBuffer> globals;
            VkDescriptorSet set0_plain = VK_NULL_HANDLE;
            VkDescriptorSet set0_shadow = VK_NULL_HANDLE;
        };

        void ensure_slot_resources(int slot, size_t draw_count);

        std::shared_ptr<VulkanDevice> device_;

        glm::vec3 light_pos_;
        bool with_shadows_;
        std::unique_ptr<ShadowPass> shadow_pass_;

        std::map<std::string, DrawableEntry> drawables_;
        // a removed drawable may still be referenced by an in-flight frame:
        // it is destroyed FRAME_SLOTS frames later
        RetireQueue<AbstractDrawable> retired_;

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
