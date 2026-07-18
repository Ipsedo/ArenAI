//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_HUD_DRAWABLES_H
#define ARENAI_VK_HUD_DRAWABLES_H

#include <functional>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_view/hud.h>

#include "../buffer.h"
#include "../descriptors.h"
#include "../device.h"
#include "../vk.h"

namespace arenai::view {

    // Everything a HUD drawable needs from the renderer that draws it: the
    // command buffer of the pass being recorded (the post-composite overlay
    // pass on the swapchain image) and the resources for lazy setup.
    struct HudFrame {
        VkCommandBuffer cmd;
        VkFormat color_format;
        std::shared_ptr<VulkanDevice> device;
        VkCommandPool upload_pool;
        DescriptorAllocator *descriptors;
    };

    class AbstractHudFrameProvider {
    public:
        virtual ~AbstractHudFrameProvider() = default;
        virtual HudFrame hud_frame() = 0;
    };

    // Internal base: the player renderer attaches itself in add_hud_drawable.
    class VulkanHudDrawable : public AbstractHudDrawable {
    public:
        void attach(AbstractHudFrameProvider *provider);

    protected:
        AbstractHudFrameProvider *provider_ = nullptr;
    };

    // Shared machinery of the line-loop HUD widgets: a "simple" color
    // pipeline (line strip, no depth) plus one vertex buffer per shape, the
    // loops closed by repeating their first point (Vulkan has no LINE_LOOP).
    class HudLineDrawable : public VulkanHudDrawable {
    protected:
        void ensure_resources();
        // draws one closed loop; mvp = vp * model, width in pixels
        void record_loop(
            const HudFrame &frame, const VulkanBuffer &loop, int nb_points,
            const glm::mat4 &mvp_matrix, float line_width) const;
        std::unique_ptr<VulkanBuffer>
        make_loop_buffer(const HudFrame &frame, const std::vector<float> &points) const;

        virtual ~HudLineDrawable();

    private:
        std::unique_ptr<HostVisibleBuffer> material_;
        VkDescriptorSetLayout empty_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout material_layout_ = VK_NULL_HANDLE;
        VkDescriptorSet material_set_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;
        std::shared_ptr<VulkanDevice> device_;
    };

    class VulkanButtonDrawable final : public HudLineDrawable {
    public:
        VulkanButtonDrawable(
            std::function<controller::button(void)> get_input, glm::vec2 center_px, float size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::button(void)> get_input_;

        std::unique_ptr<VulkanBuffer> circle_;

        float center_x_, center_y_;
        float size_;

        int nb_points_;
    };

    class VulkanJoyStickDrawable final : public HudLineDrawable {
    public:
        VulkanJoyStickDrawable(
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::joystick(void)> get_input_;

        std::unique_ptr<VulkanBuffer> square_;
        std::unique_ptr<VulkanBuffer> circle_;

        float center_x_, center_y_;
        float size_, stick_size_;

        int nb_point_bound_, nb_point_stick_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_HUD_DRAWABLES_H
