#include "./player_renderer.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace arenai::view {

    namespace {
        std::shared_ptr<const ScenePipelines> make_scene_pipelines(
            const std::shared_ptr<VulkanDevice> &device, const PlayerRendererSettings &settings) {
            return std::make_shared<const ScenePipelines>(
                device, SceneFormat{
                            .color = VK_FORMAT_R8G8B8A8_UNORM,
                            .depth = device->find_depth_format(true),
                            .samples = device->clamp_sample_count(settings.msaa_samples)});
        }
    }// namespace

    VulkanPlayerRenderer::VulkanPlayerRenderer(
        const std::shared_ptr<VulkanDevice> &device,
        std::shared_ptr<WindowFrameContext> frame_context, const int width, const int height,
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
        const PlayerRendererSettings &settings)
        : VulkanRenderer(
            device, make_scene_pipelines(device, settings), light_pos, camera, settings.shadows,
            ShadowSettings{.map_size = settings.shadow_map_size}),
          width_(width), height_(height),
          render_scale_(std::clamp(settings.render_scale, 0.5f, 1.f)),
          post_process_settings_{
              .ambient_occlusion = settings.ambient_occlusion,
              .bloom = settings.bloom,
              .sun_rays = settings.sun_rays,
              .brightness = std::clamp(settings.brightness, -1.f, 1.f)},
          frame_context_(std::move(frame_context)) {}

    int VulkanPlayerRenderer::render_width() const {
        return std::max(
            1, static_cast<int>(std::lround(static_cast<float>(width_) * render_scale_)));
    }

    int VulkanPlayerRenderer::render_height() const {
        return std::max(
            1, static_cast<int>(std::lround(static_cast<float>(height_) * render_scale_)));
    }

    void VulkanPlayerRenderer::ensure_post_process() {
        if (post_process_) return;
        post_process_ = std::make_unique<VulkanPostProcess>(
            device(), &descriptors(), render_width(), render_height(),
            make_default_post_processing_effects(
                device(), &descriptors(), render_width(), render_height(), post_process_settings_),
            pipelines().format());
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
        const VkCommandBuffer &cmd = scene_frame().cmd;

        const glm::vec3 sun_dir_view =
            glm::normalize(glm::mat3(view_matrix) * glm::normalize(light_position()));
        post_process_->run_effects(cmd, proj_matrix, sun_dir_view);

        frame_context_->begin_swapchain_pass(false, false);
        post_process_->composite_within(
            cmd, frame_context_->swapchain_format(), frame_context_->width(),
            frame_context_->height(), frame_context_->samples());

        frame_context_->end_swapchain_pass();
    }

    int VulkanPlayerRenderer::get_width() const { return width_; }

    int VulkanPlayerRenderer::get_height() const { return height_; }

    glm::mat4 VulkanPlayerRenderer::last_view_projection() const { return last_view_proj_matrix(); }

    void VulkanPlayerRenderer::set_window_size(const int new_width, const int new_height) {
        width_ = new_width;
        height_ = new_height;

        frame_context_->handle_resize();
        if (post_process_) post_process_->resize(render_width(), render_height());
    }

    VulkanPlayerRenderer::~VulkanPlayerRenderer() {
        VulkanRenderer::device()->wait_idle();
        post_process_.reset();
    }

}// namespace arenai::view
