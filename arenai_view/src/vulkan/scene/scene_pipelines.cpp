#include "./scene_pipelines.h"

#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "../core/descriptors.h"
#include "../core/pipeline.h"

namespace arenai::view {

    namespace {
        // mvp then mv
        constexpr uint32_t SCENE_PUSH_SIZE = 2 * sizeof(glm::mat4);

        const std::vector<VkVertexInputBindingDescription> POSITION_BINDING = {
            {.binding = 0, .stride = 3 * sizeof(float), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
        const std::vector<VkVertexInputAttributeDescription> POSITION_ATTRIBUTE = {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0}};

        VkPushConstantRange vertex_push(const uint32_t size) {
            return {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = size};
        }
    }// namespace

    ScenePipeline::ScenePipeline(
        std::shared_ptr<VulkanDevice> device, const VkPipelineLayout layout,
        const VkPipeline pipeline)
        : device_(std::move(device)), layout_(layout), pipeline_(pipeline) {}

    void ScenePipeline::bind(
        const VkCommandBuffer cmd, const std::span<const VkDescriptorSet> sets,
        const std::span<const uint32_t> dynamic_offsets) const {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        if (!sets.empty())
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0,
                static_cast<uint32_t>(sets.size()), sets.data(),
                static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data());
    }

    void ScenePipeline::push(
        const VkCommandBuffer cmd, const std::span<const std::byte> constants) const {
        vkCmdPushConstants(
            cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(constants.size()),
            constants.data());
    }

    ScenePipeline::~ScenePipeline() {
        vkDestroyPipeline(device_->handle(), pipeline_, nullptr);
        vkDestroyPipelineLayout(device_->handle(), layout_, nullptr);
    }

    ScenePipelines::ScenePipelines(std::shared_ptr<VulkanDevice> device, const SceneFormat &format)
        : device_(std::move(device)), format_(format) {
        const VkDevice handle = device_->handle();

        frame_layout_ = DescriptorLayoutBuilder()
                            .add_binding(
                                0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                            .build(handle);
        shadowed_frame_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(
                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                .add_binding(
                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT)
                .add_binding(
                    2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(handle);
        material_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(handle);
        sky_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(
                    0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(handle);

        const auto make = [&](const VkPipelineLayout layout, const PipelineBuilder &builder) {
            return std::make_unique<ScenePipeline>(device_, layout, builder.build(device_, layout));
        };

        diffuse_ = make(
            make_pipeline_layout(
                handle, {frame_layout_, material_layout_}, {vertex_push(SCENE_PUSH_SIZE)}),
            PipelineBuilder()
                .shaders("diffuse_vs.glsl", "diffuse_fs.glsl")
                .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                .color_format(format_.color)
                .depth_format(format_.depth)
                .samples(format_.samples));

        diffuse_shadowed_ = make(
            make_pipeline_layout(
                handle, {shadowed_frame_layout_, material_layout_}, {vertex_push(SCENE_PUSH_SIZE)}),
            PipelineBuilder()
                .shaders("diffuse_shadow_vs.glsl", "diffuse_shadow_fs.glsl")
                .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                .color_format(format_.color)
                .depth_format(format_.depth)
                .samples(format_.samples));

        diffuse_depth_ = make(
            make_pipeline_layout(handle, {}, {vertex_push(sizeof(glm::mat4))}),
            PipelineBuilder()
                .shaders("shadow_depth_vs.glsl", "shadow_depth_fs.glsl")
                .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                .depth_bias(4.f, 2.f)
                .depth_format(device_->find_depth_format(true)));

        sky_ = make(
            make_pipeline_layout(
                handle, {frame_layout_, sky_layout_}, {vertex_push(sizeof(glm::mat4))}),
            PipelineBuilder()
                .shaders("cube_vs.glsl", "cube_fs.glsl")
                .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                .color_format(format_.color)
                .depth_format(format_.depth)
                .samples(format_.samples));
    }

    const SceneFormat &ScenePipelines::format() const { return format_; }

    VkDescriptorSetLayout ScenePipelines::frame_layout() const { return frame_layout_; }

    VkDescriptorSetLayout ScenePipelines::shadowed_frame_layout() const {
        return shadowed_frame_layout_;
    }

    VkDescriptorSetLayout ScenePipelines::material_layout() const { return material_layout_; }

    VkDescriptorSetLayout ScenePipelines::sky_layout() const { return sky_layout_; }

    const ScenePipeline &ScenePipelines::diffuse() const { return *diffuse_; }

    const ScenePipeline &ScenePipelines::diffuse_shadowed() const { return *diffuse_shadowed_; }

    const ScenePipeline &ScenePipelines::diffuse_depth() const { return *diffuse_depth_; }

    const ScenePipeline &ScenePipelines::sky() const { return *sky_; }

    ScenePipelines::~ScenePipelines() {
        diffuse_.reset();
        diffuse_shadowed_.reset();
        diffuse_depth_.reset();
        sky_.reset();
        const VkDevice handle = device_->handle();
        vkDestroyDescriptorSetLayout(handle, sky_layout_, nullptr);
        vkDestroyDescriptorSetLayout(handle, material_layout_, nullptr);
        vkDestroyDescriptorSetLayout(handle, shadowed_frame_layout_, nullptr);
        vkDestroyDescriptorSetLayout(handle, frame_layout_, nullptr);
    }

}// namespace arenai::view
