#include "./cube_texture.h"

#include <utility>

#include "../../core/descriptors.h"

namespace arenai::view {

    VulkanCubeTexture::VulkanCubeTexture(std::unique_ptr<VulkanTexture> texture)
        : texture_(std::move(texture)) {}

    void VulkanCubeTexture::write_descriptor(
        const VkDevice device, const VkDescriptorSet set, const uint32_t binding) const {
        write_image_descriptor(
            device, set, binding, texture_->sampler(), texture_->view(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

}// namespace arenai::view
