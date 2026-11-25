#ifndef ARENAI_VK_PLAYER_RENDERER_H
#define ARENAI_VK_PLAYER_RENDERER_H

#include <memory>
#include <vector>

#include <arenai_view/backend.h>

#include "../../post/post_process.h"
#include "../../present/window_frame.h"
#include "./renderer.h"

namespace arenai::view {

    class VulkanPlayerRenderer final : public VulkanRenderer, public AbstractPlayerRenderer {
    public:
        VulkanPlayerRenderer(
            const std::shared_ptr<VulkanDevice> &device,
            std::shared_ptr<WindowFrameContext> frame_context, int width, int height,
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
            const PlayerRendererSettings &settings);

        int get_width() const override;
        int get_height() const override;

        void set_window_size(int new_width, int new_height) override;
        glm::mat4 last_view_projection() const override;

        VkFormat scene_color_format() const override;
        VkFormat scene_depth_format() const override;
        VkSampleCountFlagBits scene_samples() const override;

        ~VulkanPlayerRenderer() override;

    protected:
        std::pair<VkCommandBuffer, int> on_begin_frame() override;
        void on_begin_scene_pass() override;
        void on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) override;

    private:
        void ensure_post_process();
        // the scene resolution, the window size times the render scale
        int render_width() const;
        int render_height() const;

        int width_;
        int height_;
        int msaa_samples_;
        float render_scale_;
        PostProcessSettings post_process_settings_;

        std::shared_ptr<WindowFrameContext> frame_context_;

        std::unique_ptr<VulkanPostProcess> post_process_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_PLAYER_RENDERER_H
