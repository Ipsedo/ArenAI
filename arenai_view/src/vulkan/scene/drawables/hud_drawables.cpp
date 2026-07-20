//
// Created by samuel on 17/07/2026.
//

#include "./hud_drawables.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "../../core/pipeline.h"

using namespace arenai;

namespace arenai::view {

    namespace {
        constexpr glm::vec4 HUD_COLOR(1.f, 0.f, 0.f, 1.f);

        // closed loop: the first point is repeated at the end (LINE_STRIP)
        std::vector<float> get_circle_points_(const int nb_points) {
            std::vector<float> circle_points{};
            for (int i = 0; i <= nb_points; i++) {
                const double angle =
                    static_cast<double>(i % nb_points) * M_PI * 2. / static_cast<double>(nb_points);

                circle_points.push_back(static_cast<float>(cos(angle)));
                circle_points.push_back(static_cast<float>(sin(angle)));
                circle_points.push_back(0.f);
            }

            return circle_points;
        }

        glm::mat4 hud_view_projection(const float ratio) {
            const glm::mat4 v_matrix = glm::lookAt(
                glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
            const glm::mat4 p_matrix = glm::ortho(-1.f * ratio, 1.f * ratio, -1.f, 1.f, -1.f, 1.f);
            return p_matrix * v_matrix;
        }
    }// namespace

    /*
     * VulkanHudDrawable
     */

    void VulkanHudDrawable::attach(AbstractHudFrameProvider *provider) { provider_ = provider; }

    /*
     * HudLineDrawable
     */

    void HudLineDrawable::ensure_resources() {
        if (pipeline_ != VK_NULL_HANDLE) return;
        const HudFrame frame = provider_->hud_frame();
        device_ = frame.device;

        material_ = std::make_unique<HostVisibleBuffer>(
            device_, sizeof(glm::vec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(material_->data(), &HUD_COLOR, sizeof(glm::vec4));
        material_->flush();

        // the simple shaders read only set 1 (material): set 0 stays an empty
        // placeholder layout, never bound
        empty_layout_ = DescriptorLayoutBuilder().build(device_->handle());
        material_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(device_->handle());
        material_set_ = frame.descriptors->allocate(material_layout_);
        write_buffer_descriptor(
            device_->handle(), material_set_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            material_->handle(), 0, sizeof(glm::vec4));

        pipeline_layout_ = make_pipeline_layout(
            device_->handle(), {empty_layout_, material_layout_},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)}});
        pipeline_ = PipelineBuilder()
                        .shaders("simple_vs.glsl", "simple_fs.glsl")
                        .vertex_input(
                            {{0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}},
                            {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}})
                        .topology(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
                        .dynamic_line_width()
                        .cull_mode(VK_CULL_MODE_NONE)
                        .depth(false, false)
                        .color_format(frame.color_format)
                        .build(device_, pipeline_layout_);
    }

    std::unique_ptr<VulkanBuffer> HudLineDrawable::make_loop_buffer(
        const HudFrame &frame, const std::vector<float> &points) const {
        return std::make_unique<VulkanBuffer>(
            frame.device, frame.upload_pool, points.data(), points.size() * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    void HudLineDrawable::record_loop(
        const HudFrame &frame, const VulkanBuffer &loop, const int nb_points,
        const glm::mat4 &mvp_matrix, const float line_width) const {
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(
            frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 1, 1, &material_set_, 0,
            nullptr);
        vkCmdPushConstants(
            frame.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
            &mvp_matrix);
        vkCmdSetLineWidth(frame.cmd, frame.device->wide_lines() ? line_width : 1.f);

        const VkBuffer vertex_buffer = loop.handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);
        // +1: the loop is closed by the repeated first point
        vkCmdDraw(frame.cmd, nb_points + 1, 1, 0, 0);
    }

    HudLineDrawable::~HudLineDrawable() {
        if (!device_) return;
        vkDestroyPipeline(device_->handle(), pipeline_, nullptr);
        vkDestroyPipelineLayout(device_->handle(), pipeline_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_->handle(), material_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_->handle(), empty_layout_, nullptr);
    }

    /*
     * VulkanButtonDrawable
     */

    VulkanButtonDrawable::VulkanButtonDrawable(
        std::function<controller::button(void)> get_input, const glm::vec2 center_px,
        const float size_px)
        : get_input_(std::move(get_input)), center_x_(center_px.x), center_y_(center_px.y),
          size_(size_px), nb_points_(128) {}

    void VulkanButtonDrawable::draw(const int width, const int height) {
        ensure_resources();
        const HudFrame frame = provider_->hud_frame();
        if (!circle_) circle_ = make_loop_buffer(frame, get_circle_points_(nb_points_));

        const float ratio = static_cast<float>(width) / static_cast<float>(height);

        const float center_x_rel = ratio * (center_x_ / static_cast<float>(width) * 2.f - 1.f);
        const float center_y_rel = center_y_ / static_cast<float>(height) * 2.f - 1.f;

        auto [pressed] = get_input_();

        const float size_rel = size_ / static_cast<float>(width) * ratio;

        const glm::mat4 vp_matrix = hud_view_projection(ratio);
        const glm::mat4 button_m_matrix = glm::translate(glm::vec3(center_x_rel, center_y_rel, 0.f))
                                          * glm::scale(glm::vec3(size_rel, size_rel, 1.f));

        record_loop(frame, *circle_, nb_points_, vp_matrix * button_m_matrix, pressed ? 8.f : 5.f);
    }

    /*
     * VulkanJoyStickDrawable
     */

    VulkanJoyStickDrawable::VulkanJoyStickDrawable(
        std::function<controller::joystick(void)> get_input_px, const glm::vec2 center_px,
        const float size_px, const float stick_size_px)
        : get_input_(std::move(get_input_px)), center_x_(center_px.x), center_y_(center_px.y),
          size_(size_px), stick_size_(stick_size_px), nb_point_bound_(4), nb_point_stick_(128) {}

    void VulkanJoyStickDrawable::draw(const int width, const int height) {
        ensure_resources();
        const HudFrame frame = provider_->hud_frame();
        if (!square_)
            square_ = make_loop_buffer(
                frame,
                {-1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, -1.f, 0.f, -1.f, -1.f, 0.f, -1.f, 1.f, 0.f});
        if (!circle_) circle_ = make_loop_buffer(frame, get_circle_points_(nb_point_stick_));

        const float ratio = static_cast<float>(width) / static_cast<float>(height);

        const float center_x_rel = ratio * (center_x_ / static_cast<float>(width) * 2.f - 1.f);
        const float center_y_rel = center_y_ / static_cast<float>(height) * 2.f - 1.f;

        auto [stick_x, stick_y] = get_input_();
        const float stick_x_rel = ratio * (stick_x / static_cast<float>(width) * 2.f - 1.f);
        const float stick_y_rel = stick_y / static_cast<float>(height) * 2.f - 1.f;

        const float size_rel = size_ / static_cast<float>(width) * ratio;
        const float stick_size_rel = stick_size_ / static_cast<float>(width) * ratio;

        const glm::mat4 vp_matrix = hud_view_projection(ratio);

        const glm::mat4 bounds_m_matrix = glm::translate(glm::vec3(center_x_rel, center_y_rel, 0.f))
                                          * glm::scale(glm::vec3(size_rel, size_rel, 1.f));
        const glm::mat4 stick_m_matrix =
            glm::translate(glm::vec3(stick_x_rel, stick_y_rel, 0.f))
            * glm::scale(glm::vec3(stick_size_rel, stick_size_rel, 1.f));

        record_loop(frame, *square_, nb_point_bound_, vp_matrix * bounds_m_matrix, 5.f);
        record_loop(frame, *circle_, nb_point_stick_, vp_matrix * stick_m_matrix, 5.f);
    }

}// namespace arenai::view
