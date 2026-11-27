#ifndef ARENAI_VK_SCENE_PIPELINES_H
#define ARENAI_VK_SCENE_PIPELINES_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "../core/device.h"
#include "../core/vk.h"

namespace arenai::view {

    struct SceneFormat {
        VkFormat color;
        VkFormat depth;
        VkSampleCountFlagBits samples;
    };

    class ScenePipeline {
    public:
        ScenePipeline(
            std::shared_ptr<VulkanDevice> device, VkPipelineLayout layout, VkPipeline pipeline);

        ScenePipeline(const ScenePipeline &) = delete;
        ScenePipeline &operator=(const ScenePipeline &) = delete;

        void bind(
            VkCommandBuffer cmd, std::span<const VkDescriptorSet> sets,
            std::span<const uint32_t> dynamic_offsets) const;
        void push(VkCommandBuffer cmd, std::span<const std::byte> constants) const;

        ~ScenePipeline();

    private:
        std::shared_ptr<VulkanDevice> device_;
        VkPipelineLayout layout_;
        VkPipeline pipeline_;
    };

    class ScenePipelines {
    public:
        ScenePipelines(std::shared_ptr<VulkanDevice> device, const SceneFormat &format);

        ScenePipelines(const ScenePipelines &) = delete;
        ScenePipelines &operator=(const ScenePipelines &) = delete;

        const SceneFormat &format() const;

        // set layouts drawables and renderers allocate from
        VkDescriptorSetLayout frame_layout() const;
        VkDescriptorSetLayout shadowed_frame_layout() const;
        VkDescriptorSetLayout material_layout() const;
        VkDescriptorSetLayout sky_layout() const;

        const ScenePipeline &diffuse() const;
        const ScenePipeline &diffuse_shadowed() const;
        const ScenePipeline &diffuse_depth() const;
        const ScenePipeline &sky() const;

        ~ScenePipelines();

    private:
        std::shared_ptr<VulkanDevice> device_;
        SceneFormat format_;

        VkDescriptorSetLayout frame_layout_;
        VkDescriptorSetLayout shadowed_frame_layout_;
        VkDescriptorSetLayout material_layout_;
        VkDescriptorSetLayout sky_layout_;

        std::unique_ptr<ScenePipeline> diffuse_;
        std::unique_ptr<ScenePipeline> diffuse_shadowed_;
        std::unique_ptr<ScenePipeline> diffuse_depth_;
        std::unique_ptr<ScenePipeline> sky_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SCENE_PIPELINES_H
