//
// Created by samuel on 17/07/2026.
//

#include "./descriptors.h"

#include <utility>

#include "./errors.h"

namespace arenai::view {

    /*
     * DescriptorLayoutBuilder
     */

    DescriptorLayoutBuilder &DescriptorLayoutBuilder::add_binding(
        const uint32_t binding, const VkDescriptorType type, const VkShaderStageFlags stages) {
        VkDescriptorSetLayoutBinding layout_binding{};
        layout_binding.binding = binding;
        layout_binding.descriptorType = type;
        layout_binding.descriptorCount = 1;
        layout_binding.stageFlags = stages;
        bindings_.push_back(layout_binding);
        return *this;
    }

    VkDescriptorSetLayout DescriptorLayoutBuilder::build(const VkDevice device) const {
        VkDescriptorSetLayoutCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = static_cast<uint32_t>(bindings_.size());
        create_info.pBindings = bindings_.data();

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        vk_check(
            vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout),
            "vkCreateDescriptorSetLayout");
        return layout;
    }

    /*
     * DescriptorAllocator
     */

    DescriptorAllocator::DescriptorAllocator(std::shared_ptr<VulkanDevice> device)
        : device_(std::move(device)) {}

    VkDescriptorPool DescriptorAllocator::make_pool() const {
        constexpr uint32_t sets_per_pool = 64;
        const std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * sets_per_pool},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, sets_per_pool},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * sets_per_pool},
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = sets_per_pool;
        pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
        pool_info.pPoolSizes = sizes.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;
        vk_check(
            vkCreateDescriptorPool(device_->handle(), &pool_info, nullptr, &pool),
            "vkCreateDescriptorPool");
        return pool;
    }

    VkDescriptorSet DescriptorAllocator::allocate(const VkDescriptorSetLayout layout) {
        if (pools_.empty()) pools_.push_back(make_pool());

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pools_.back();
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &layout;

        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult result = vkAllocateDescriptorSets(device_->handle(), &alloc_info, &set);
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
            pools_.push_back(make_pool());
            alloc_info.descriptorPool = pools_.back();
            result = vkAllocateDescriptorSets(device_->handle(), &alloc_info, &set);
        }
        vk_check(result, "vkAllocateDescriptorSets");
        return set;
    }

    DescriptorAllocator::~DescriptorAllocator() {
        for (const auto pool: pools_) vkDestroyDescriptorPool(device_->handle(), pool, nullptr);
    }

    /*
     * Write helpers
     */

    void write_buffer_descriptor(
        const VkDevice device, const VkDescriptorSet set, const uint32_t binding,
        const VkDescriptorType type, const VkBuffer buffer, const VkDeviceSize offset,
        const VkDeviceSize range) {
        VkDescriptorBufferInfo buffer_info{buffer, offset, range};

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = type;
        write.pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void write_image_descriptor(
        const VkDevice device, const VkDescriptorSet set, const uint32_t binding,
        const VkSampler sampler, const VkImageView view, const VkImageLayout layout) {
        VkDescriptorImageInfo image_info{sampler, view, layout};

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image_info;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

}// namespace arenai::view
