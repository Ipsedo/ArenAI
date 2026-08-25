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

        void make_current() const override;
        void release_current() const override;

        /*
         * DrawableContext port
         */

        const SceneFrame &scene_frame() const override;
        VkCommandPool upload_pool() const override;
        VkFormat shadow_depth_format() const override;

        const std::shared_ptr<VulkanDevice> &device() const override;

        DescriptorAllocator &descriptors() override;

        VkDescriptorSetLayout set0_plain_layout() const override;
        VkDescriptorSetLayout set0_shadow_layout() const override;

    protected:
        virtual std::pair<VkCommandBuffer, int> on_begin_frame() = 0;

        virtual void on_begin_scene_pass() = 0;

        virtual void on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) = 0;

        const glm::vec3 &light_position() const;
        const std::shared_ptr<AbstractCamera> &camera() const;
        const glm::mat4 &last_view_proj_matrix() const;

    private:
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

        RetireQueue<AbstractDrawable> retired_;

        std::shared_ptr<AbstractCamera> camera_;

        VkCommandPool upload_pool_;
        std::unique_ptr<DescriptorAllocator> descriptors_;
        VkDescriptorSetLayout set0_plain_layout_;
        VkDescriptorSetLayout set0_shadow_layout_;

        SlotResources slots_[FRAME_SLOTS];
        SceneFrame frame_;
        glm::mat4 last_view_proj_matrix_{1.f};
    };

}// namespace arenai::view

#endif// ARENAI_VK_RENDERER_H
