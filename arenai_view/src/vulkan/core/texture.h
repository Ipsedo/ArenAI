#ifndef ARENAI_VK_TEXTURE_H
#define ARENAI_VK_TEXTURE_H

#include <filesystem>
#include <memory>
#include <vector>

#include <arenai_utils/file_reader.h>

#include "./device.h"
#include "./vk.h"
#include "./vma.h"

namespace arenai::view {

    class VulkanTexture {
    public:
        VulkanTexture(
            const std::shared_ptr<VulkanDevice> &device, const VkCommandPool &pool, int width,
            int height, int channels, const uint8_t *pixels);

        static std::unique_ptr<VulkanTexture>
        make_white(const std::shared_ptr<VulkanDevice> &device, const VkCommandPool &pool);

        static std::unique_ptr<VulkanTexture> from_png(
            const std::shared_ptr<VulkanDevice> &device, const VkCommandPool &pool,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &png_path);

        static std::unique_ptr<VulkanTexture> cube_from_pngs(
            const std::shared_ptr<VulkanDevice> &device, const VkCommandPool &pool,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path);

        VulkanTexture(const VulkanTexture &) = delete;
        VulkanTexture &operator=(const VulkanTexture &) = delete;

        VkImageView view() const;
        VkSampler sampler() const;

        ~VulkanTexture();

    private:
        // cube = 6 layers, pixels_per_face laid out +X,-X,+Y,-Y,+Z,-Z
        VulkanTexture(
            const std::shared_ptr<VulkanDevice> &device, const VkCommandPool &pool, int width,
            int height, const std::vector<std::vector<uint8_t>> &rgba_layers, bool cube);

        void upload(
            const VkCommandPool &pool, const std::vector<std::vector<uint8_t>> &rgba_layers) const;
        void record_layout_transition(
            const VkCommandBuffer &cmd, VkImageLayout old_layout, VkImageLayout new_layout) const;

        std::shared_ptr<VulkanDevice> device_;
        VkImage image_;
        VmaAllocation allocation_;
        VkImageView view_;
        VkSampler sampler_;
        int width_, height_;
        uint32_t layers_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_TEXTURE_H
