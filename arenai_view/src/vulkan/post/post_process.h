//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_POST_PROCESS_H
#define ARENAI_VK_POST_PROCESS_H

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "../core/descriptors.h"
#include "../core/device.h"
#include "../core/render_target.h"
#include "./effect.h"

namespace arenai::view {

    // Player-only post-processing pipeline: the scene is rendered into a 4x
    // MSAA target, resolved (color + depth) through the dynamic-rendering
    // resolve attachments, then run through the ordered effect chain. The
    // final composite pass is recorded separately, inside the rendering
    // scope the caller opened on its output (swapchain image or test target),
    // so the HUD can share that scope.
    class VulkanPostProcess {
    public:
        VulkanPostProcess(
            std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors, int width,
            int height, std::vector<std::shared_ptr<VulkanPostEffect>> ordered_effects);

        VulkanPostProcess(const VulkanPostProcess &) = delete;
        VulkanPostProcess &operator=(const VulkanPostProcess &) = delete;

        void resize(int new_width, int new_height);

        VkFormat scene_color_format() const;
        VkFormat scene_depth_format() const;
        VkSampleCountFlagBits scene_samples() const;

        // begins the MSAA scene rendering scope (with the resolve
        // attachments) and sets the negative-height scene viewport
        void begin_scene_pass(VkCommandBuffer cmd);

        // ends the scene scope and runs every effect except the final
        // composite; proj_matrix is the scene projection (depth
        // reconstruction) and sun_dir_view the normalized view-space
        // direction toward the sun
        void run_effects(
            VkCommandBuffer cmd, const glm::mat4 &proj_matrix, const glm::vec3 &sun_dir_view);

        // records the composite draw inside the caller's open rendering scope
        void composite_within(
            VkCommandBuffer cmd, VkFormat output_format, int output_width, int output_height);

        ~VulkanPostProcess() = default;

    private:
        static constexpr int MSAA_SAMPLES = 4;

        void create_scene_targets();

        std::shared_ptr<VulkanDevice> device_;
        DescriptorAllocator *descriptors_;

        int width_;
        int height_;

        // frame counter animating the film grain (wraps to stay float-exact)
        int frame_;

        VkSampleCountFlagBits samples_;
        VkFormat depth_format_;

        std::unique_ptr<Target> msaa_color_;
        std::unique_ptr<Target> msaa_depth_;
        std::unique_ptr<Target> resolve_color_;
        std::unique_ptr<Target> resolve_depth_;

        std::vector<std::shared_ptr<VulkanPostEffect>> ordered_effects_;

        VulkanPostEffect::FrameContext context_{};
    };

    // the standard player chain: SSAO → AO blur → bloom bright → bloom blur
    // → god rays → composite
    std::vector<std::shared_ptr<VulkanPostEffect>> make_default_post_processing_effects(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors, int width,
        int height);

}// namespace arenai::view

#endif// ARENAI_VK_POST_PROCESS_H
