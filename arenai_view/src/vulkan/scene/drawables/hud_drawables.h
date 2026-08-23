//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_HUD_DRAWABLES_H
#define ARENAI_VK_HUD_DRAWABLES_H

#include <functional>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_view/hud.h>

#include "../../core/buffer.h"
#include "../../core/descriptors.h"
#include "../../core/device.h"
#include "../../core/vk.h"

namespace arenai::view {

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

    class VulkanHudDrawable : public AbstractHudDrawable {
    public:
        void attach(AbstractHudFrameProvider *provider);

    protected:
        AbstractHudFrameProvider *provider_ = nullptr;
    };

    class HudLineDrawable : public VulkanHudDrawable {
    protected:
        void ensure_resources();

        void record_loop(
            const HudFrame &frame, const VulkanBuffer &loop, int nb_points,
            const glm::mat4 &mvp_matrix, float line_width) const;

        static std::unique_ptr<VulkanBuffer>
        make_loop_buffer(const HudFrame &frame, const std::vector<float> &points);

        ~HudLineDrawable() override;

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
            std::function<controller::button()> get_input, glm::vec2 center_px, float size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::button()> get_input_;

        std::unique_ptr<VulkanBuffer> circle_;

        float center_x_, center_y_;
        float size_;

        int nb_points_;
    };

    class VulkanJoyStickDrawable final : public HudLineDrawable {
    public:
        VulkanJoyStickDrawable(
            std::function<controller::joystick()> get_input_px, glm::vec2 center_px, float size_px,
            float stick_size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::joystick()> get_input_;

        std::unique_ptr<VulkanBuffer> square_;
        std::unique_ptr<VulkanBuffer> circle_;

        float center_x_, center_y_;
        float size_, stick_size_;

        int nb_point_bound_, nb_point_stick_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_HUD_DRAWABLES_H
