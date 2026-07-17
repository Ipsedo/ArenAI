//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_OFFSCREEN_RENDERER_H
#define ARENAI_VK_OFFSCREEN_RENDERER_H

#include <memory>

#include "../render_target.h"
#include "./renderer.h"

namespace arenai::view {

    // Offscreen renderer for the agents' vision. Keeps the asynchronous
    // 2-slot readback of the GL PBO implementation: frame N is submitted
    // without waiting and the frame N-1 pixels are returned, the very first
    // call returning a black image — arenai_core's 2-frame warm-up relies on
    // this exact latency.
    class VulkanOffscreenRenderer final : public VulkanRenderer, public AbstractOffscreenRenderer {
    public:
        VulkanOffscreenRenderer(
            const std::shared_ptr<VulkanDevice> &device, int width, int height, glm::vec3 light_pos,
            const std::shared_ptr<AbstractCamera> &camera, bool with_shadows = false);

        image<uint8_t> draw_and_get_frame(
            const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        int get_width() const override;
        int get_height() const override;

        VkFormat scene_color_format() const override;
        VkFormat scene_depth_format() const override;
        VkSampleCountFlagBits scene_samples() const override;

        ~VulkanOffscreenRenderer() override;

    protected:
        std::pair<VkCommandBuffer, int> on_begin_frame() override;
        void on_begin_scene_pass() override;
        void on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) override;

    private:
        struct Slot {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            std::unique_ptr<HostVisibleBuffer> readback;
            bool submitted = false;
        };

        image<uint8_t> read_slot(const Slot &slot) const;

        int width_, height_;
        VkFormat depth_format_;

        std::unique_ptr<Target> color_;
        std::unique_ptr<Target> depth_;

        Slot slots_[FRAME_SLOTS];
        int slot_index_ = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VK_OFFSCREEN_RENDERER_H
