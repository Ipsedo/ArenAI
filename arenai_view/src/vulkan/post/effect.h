//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_POST_PROCESSING_EFFECT_H
#define ARENAI_VK_POST_PROCESSING_EFFECT_H

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "../core/descriptors.h"
#include "../core/device.h"
#include "../core/render_target.h"
#include "../core/vk.h"

namespace arenai::view {

    enum class PostTexture {
        ao_raw,      // SsaoEffect -> AoBlurEffect
        ao,          // AoBlurEffect -> CompositeEffect
        bloom_bright,// BloomBrightEffect -> BloomBlurEffect
        bloom,       // BloomBlurEffect -> CompositeEffect
        god_rays,    // GodRaysEffect -> CompositeEffect
    };

    enum class PostScalar {
        god_ray_strength,// GodRaysEffect -> CompositeEffect
    };

    class VulkanPostEffect {
    public:
        struct FrameContext {
            VkCommandBuffer cmd;
            const Target *scene;
            const Target *depth;
            int screen_width;
            int screen_height;
            glm::mat4 proj_matrix;
            glm::vec4 proj_info;
            glm::vec3 sun_dir_view;
            int frame;

            std::unordered_map<PostTexture, const Target *> textures;
            std::unordered_map<PostScalar, float> scalars;

            VkFormat output_format;
            int output_width;
            int output_height;
        };

        virtual ~VulkanPostEffect();

        void resize(int new_width, int new_height);

        virtual void render(FrameContext &context) = 0;

    protected:
        struct TargetSpec {
            VkFormat format;
            int size_divisor;
        };

        VulkanPostEffect(
            std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors,
            std::string fragment_shader, uint32_t nb_inputs, uint32_t push_size,
            std::vector<TargetSpec> specs, int width, int height);

        void run_pass(
            const FrameContext &context, size_t target_index,
            const std::vector<const Target *> &inputs, const void *push_data);

        void run_inline(
            const FrameContext &context, const std::vector<const Target *> &inputs,
            const void *push_data);

        void ensure_target_readable(const FrameContext &context, size_t target_index);

        const Target *target(size_t index) const;

        std::shared_ptr<VulkanDevice> device_;

    private:
        void create_targets(int width, int height);
        VkPipeline pipeline_for(VkFormat color_format);
        VkDescriptorSet set_for(const std::vector<const Target *> &inputs);
        void record_draw(
            const FrameContext &context, VkFormat color_format,
            const std::vector<const Target *> &inputs, const void *push_data);

        DescriptorAllocator *descriptors_;
        std::string fragment_shader_;
        uint32_t nb_inputs_;
        uint32_t push_size_;
        std::vector<TargetSpec> specs_;

        std::vector<std::unique_ptr<Target>> targets_;
        std::vector<bool> target_initialized_;

        VkSampler linear_sampler_ = VK_NULL_HANDLE;
        VkSampler nearest_sampler_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout input_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout empty_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;

        std::map<VkFormat, VkPipeline> pipelines_;

        std::map<std::vector<const Target *>, VkDescriptorSet> input_sets_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_POST_PROCESSING_EFFECT_H
