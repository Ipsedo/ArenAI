#ifndef ARENAI_VK_CUBE_TEXTURE_H
#define ARENAI_VK_CUBE_TEXTURE_H

#include <memory>

#include <arenai_view/drawable.h>

#include "../../core/texture.h"
#include "../../core/vk.h"

namespace arenai::view {

    class VulkanCubeTexture final : public AbstractCubeTexture {
    public:
        explicit VulkanCubeTexture(std::unique_ptr<VulkanTexture> texture);

        void write_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding) const;

    private:
        std::unique_ptr<VulkanTexture> texture_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_CUBE_TEXTURE_H
