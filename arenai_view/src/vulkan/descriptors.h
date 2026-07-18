//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_DESCRIPTORS_H
#define ARENAI_VK_DESCRIPTORS_H

#include <memory>
#include <vector>

#include "./device.h"
#include "./vk.h"

namespace arenai::view {

    class DescriptorLayoutBuilder {
    public:
        DescriptorLayoutBuilder &
        add_binding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stages);

        VkDescriptorSetLayout build(VkDevice device) const;

    private:
        std::vector<VkDescriptorSetLayoutBinding> bindings_;
    };

    // Growable descriptor pool. Thread-confined, like the command pools: each
    // renderer (and the Rml render interface) owns its own allocator on its
    // thread. Sets live as long as the allocator, there is no per-set free.
    class DescriptorAllocator {
    public:
        explicit DescriptorAllocator(std::shared_ptr<VulkanDevice> device);

        DescriptorAllocator(const DescriptorAllocator &) = delete;
        DescriptorAllocator &operator=(const DescriptorAllocator &) = delete;

        VkDescriptorSet allocate(VkDescriptorSetLayout layout);

        ~DescriptorAllocator();

    private:
        VkDescriptorPool make_pool() const;

        std::shared_ptr<VulkanDevice> device_;
        std::vector<VkDescriptorPool> pools_;
    };

    // Descriptor write helpers (immediate vkUpdateDescriptorSets)
    void write_buffer_descriptor(
        VkDevice device, VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
        VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
    void write_image_descriptor(
        VkDevice device, VkDescriptorSet set, uint32_t binding, VkSampler sampler, VkImageView view,
        VkImageLayout layout);

}// namespace arenai::view

#endif// ARENAI_VK_DESCRIPTORS_H
