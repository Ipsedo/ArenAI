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

#include "../descriptors.h"
#include "../device.h"
#include "../render_target.h"
#include "../vk.h"

namespace arenai::view {

    // One pass (or ping-pong group of passes) of the post-processing chain.
    // The concrete effect declares its render targets (format + resolution
    // divisor) at construction; this base class owns their lifecycle, the
    // per-pass pipeline and the fullscreen-triangle draw. Effects communicate
    // through the FrameContext: each pass publishes its output target under a
    // name that later passes look up.
    //
    // The passes are pure image-space work: they rasterize with a regular
    // (positive-height) viewport so that uv maps rows identically between
    // input and output — only the scene pass uses the negative viewport.
    class VulkanPostEffect {
    public:
        struct FrameContext {
            VkCommandBuffer cmd;
            const Target *scene;// resolved scene color, SHADER_READ layout
            const Target *depth;// resolved scene depth, SHADER_READ layout
            int screen_width;
            int screen_height;
            glm::mat4 proj_matrix;
            glm::vec4 proj_info;
            glm::vec3 sun_dir_view;
            int frame;

            std::unordered_map<std::string, const Target *> textures;
            std::unordered_map<std::string, float> scalars;

            // the composite pass draws into the caller's open rendering scope
            VkFormat output_format;
            int output_width;
            int output_height;
        };

        virtual ~VulkanPostEffect();

        // recreates the effect's targets at the new screen resolution
        void resize(int new_width, int new_height);

        // records the pass(es) and publishes the effect's outputs
        virtual void render(FrameContext &context) = 0;

    protected:
        // declares one target: screen resolution / size_divisor
        struct TargetSpec {
            VkFormat format;
            int size_divisor;
        };

        VulkanPostEffect(
            std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors,
            std::string fragment_shader, uint32_t nb_inputs, uint32_t push_size,
            std::vector<TargetSpec> specs, int width, int height);

        // renders a fullscreen triangle into targets[target_index], sampling
        // the given inputs; handles the layout barriers around the target
        void run_pass(
            const FrameContext &context, size_t target_index,
            const std::vector<const Target *> &inputs, const void *push_data);

        // records the fullscreen draw inside the caller's open rendering
        // scope (composite pass): the pipeline targets context.output_format
        void run_inline(
            const FrameContext &context, const std::vector<const Target *> &inputs,
            const void *push_data);

        // puts a target in a sampleable layout even when its pass was
        // skipped this frame (god rays with the sun out of frame)
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
        // one pipeline per output format (own targets + the composite output)
        std::map<VkFormat, VkPipeline> pipelines_;
        // descriptor sets cached by input combination, cleared on resize
        std::map<std::vector<const Target *>, VkDescriptorSet> input_sets_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_POST_PROCESSING_EFFECT_H
