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

    class VulkanPostProcess {
    public:
        VulkanPostProcess(
            std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors, int width,
            int height, std::vector<std::shared_ptr<VulkanPostEffect>> ordered_effects,
            int msaa_samples = 4);

        VulkanPostProcess(const VulkanPostProcess &) = delete;
        VulkanPostProcess &operator=(const VulkanPostProcess &) = delete;

        void resize(int new_width, int new_height);

        static VkFormat scene_color_format();
        VkFormat scene_depth_format() const;
        VkSampleCountFlagBits scene_samples() const;

        void begin_scene_pass(const VkCommandBuffer &cmd) const;

        void run_effects(
            const VkCommandBuffer &cmd, const glm::mat4 &proj_matrix,
            const glm::vec3 &sun_dir_view);

        void composite_within(
            const VkCommandBuffer &cmd, VkFormat output_format, int output_width,
            int output_height);

        ~VulkanPostProcess() = default;

    private:
        void create_scene_targets();

        std::shared_ptr<VulkanDevice> device_;
        DescriptorAllocator *descriptors_;

        int width_;
        int height_;

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

    std::vector<std::shared_ptr<VulkanPostEffect>> make_default_post_processing_effects(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors, int width,
        int height);

}// namespace arenai::view

#endif// ARENAI_VK_POST_PROCESS_H
