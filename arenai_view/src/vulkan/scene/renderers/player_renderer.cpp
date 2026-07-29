//
// Created by samuel on 17/07/2026.
//

#include "./player_renderer.h"

#include <utility>

namespace arenai::view {

    VulkanPlayerRenderer::VulkanPlayerRenderer(
        const std::shared_ptr<VulkanDevice> &device,
        std::shared_ptr<WindowFrameContext> frame_context, const int width, const int height,
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera)
        : VulkanRenderer(device, light_pos, camera, true), width_(width), height_(height),
          frame_context_(std::move(frame_context)), hud_drawables_() {}

    void VulkanPlayerRenderer::add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) {
        if (auto *vulkan_hud = dynamic_cast<VulkanHudDrawable *>(hud_drawable.get()))
            vulkan_hud->attach(this);
        hud_drawables_.push_back(std::move(hud_drawable));
    }

    void VulkanPlayerRenderer::ensure_post_process() {
        if (post_process_) return;
        post_process_ = std::make_unique<VulkanPostProcess>(
            device(), &descriptors(), get_width(), get_height(),
            make_default_post_processing_effects(
                device(), &descriptors(), get_width(), get_height()));
    }

    std::pair<VkCommandBuffer, int> VulkanPlayerRenderer::on_begin_frame() {
        if (!frame_context_->ensure_frame_begun()) return {VK_NULL_HANDLE, 0};
        return {frame_context_->cmd(), frame_context_->slot()};
    }

    void VulkanPlayerRenderer::on_begin_scene_pass() {
        ensure_post_process();
        post_process_->begin_scene_pass(scene_frame().cmd);
    }

    void
    VulkanPlayerRenderer::on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) {
        const VkCommandBuffer cmd = scene_frame().cmd;

        // post-processing chain, then composite + HUD onto the swapchain
        // image; the HUD shares the composite's rendering scope so that it
        // stays untouched by the effects
        const glm::vec3 sun_dir_view =
            glm::normalize(glm::mat3(view_matrix) * glm::normalize(light_position()));
        post_process_->run_effects(cmd, proj_matrix, sun_dir_view);

        frame_context_->begin_swapchain_pass(false, false);
        post_process_->composite_within(
            cmd, frame_context_->swapchain_format(), frame_context_->width(),
            frame_context_->height());

        // the composite pass switched to its image-space viewport: restore
        // the negative-height one the HUD math expects
        const VkViewport viewport{
            0.f,
            static_cast<float>(frame_context_->height()),
            static_cast<float>(frame_context_->width()),
            -static_cast<float>(frame_context_->height()),
            0.f,
            1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        for (const auto &hud_drawable: hud_drawables_)
            hud_drawable->draw(get_width(), get_height());

        frame_context_->end_swapchain_pass();

        // no present here: presenting the frame is the application's call (an
        // UI overlay may still be drawn on top), see
        // AbstractWindowedGraphicBackend::present()
    }

    int VulkanPlayerRenderer::get_width() const { return width_; }

    int VulkanPlayerRenderer::get_height() const { return height_; }

    void VulkanPlayerRenderer::set_window_size(const int new_width, const int new_height) {
        width_ = new_width;
        height_ = new_height;

        frame_context_->handle_resize();
        if (post_process_) post_process_->resize(new_width, new_height);
    }

    VkFormat VulkanPlayerRenderer::scene_color_format() const {
        return post_process_->scene_color_format();
    }

    VkFormat VulkanPlayerRenderer::scene_depth_format() const {
        return post_process_->scene_depth_format();
    }

    VkSampleCountFlagBits VulkanPlayerRenderer::scene_samples() const {
        return post_process_->scene_samples();
    }

    HudFrame VulkanPlayerRenderer::hud_frame() {
        return {
            frame_context_->cmd(), frame_context_->swapchain_format(), device(), upload_pool(),
            &descriptors()};
    }

    VulkanPlayerRenderer::~VulkanPlayerRenderer() {
        // in-flight frames may still reference the pipelines and buffers of
        // the drawables/effects destroyed with this renderer
        device()->wait_idle();
        hud_drawables_.clear();
        post_process_.reset();
    }

}// namespace arenai::view
