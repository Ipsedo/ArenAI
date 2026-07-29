//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_PLAYER_RENDERER_H
#define ARENAI_VK_PLAYER_RENDERER_H

#include <memory>
#include <vector>

#include "../../post/post_process.h"
#include "../../present/window_frame.h"
#include "../drawables/hud_drawables.h"
#include "./renderer.h"

namespace arenai::view {

    // Player view: scene into the MSAA post-processing targets, effect
    // chain, then composite + HUD into the swapchain image, all recorded in
    // the frame command buffer owned by the shared WindowFrameContext. The
    // frame stays open after draw(): the application may still composite the
    // UI overlay before present().
    class VulkanPlayerRenderer final : public VulkanRenderer,
                                       public AbstractPlayerRenderer,
                                       public AbstractHudFrameProvider {
    public:
        VulkanPlayerRenderer(
            const std::shared_ptr<VulkanDevice> &device,
            std::shared_ptr<WindowFrameContext> frame_context, int width, int height,
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera);

        void add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) override;

        int get_width() const override;
        int get_height() const override;

        void set_window_size(int new_width, int new_height) override;

        VkFormat scene_color_format() const override;
        VkFormat scene_depth_format() const override;
        VkSampleCountFlagBits scene_samples() const override;

        HudFrame hud_frame() override;

        ~VulkanPlayerRenderer() override;

    protected:
        std::pair<VkCommandBuffer, int> on_begin_frame() override;
        void on_begin_scene_pass() override;
        void on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) override;

    private:
        void ensure_post_process();

        int width_;
        int height_;

        std::shared_ptr<WindowFrameContext> frame_context_;

        // MSAA + tonemapping/grading pipeline, built lazily on first frame
        std::unique_ptr<VulkanPostProcess> post_process_;

        std::vector<std::unique_ptr<AbstractHudDrawable>> hud_drawables_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_PLAYER_RENDERER_H
