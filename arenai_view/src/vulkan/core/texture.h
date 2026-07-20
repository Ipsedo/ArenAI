//
// Created by samuel on 17/07/2026.
//

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

    // Sampled R8G8B8A8_UNORM texture (2D or cube), uploaded once at creation
    // on the caller's thread and left in SHADER_READ_ONLY layout. RGB images
    // are expanded to RGBA on the CPU (tightly-packed RGB formats have no
    // guaranteed sampling support in Vulkan).
    class VulkanTexture {
    public:
        // 2D texture from raw RGBA/RGB bytes
        VulkanTexture(
            const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool, int width, int height,
            int channels, const uint8_t *pixels);

        // 1x1 opaque white texture (untextured UI geometry)
        static std::unique_ptr<VulkanTexture>
        make_white(const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool);

        // 2D texture from a png resource
        static std::unique_ptr<VulkanTexture> from_png(
            const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &png_path);

        // cube map from the 6 pngs (posx/negx/posy/negy/posz/negz.png) of a folder
        static std::unique_ptr<VulkanTexture> cube_from_pngs(
            const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path);

        VulkanTexture(const VulkanTexture &) = delete;
        VulkanTexture &operator=(const VulkanTexture &) = delete;

        VkImageView view() const;
        // linear filtering, clamp to edge (parity with the GL texture setup)
        VkSampler sampler() const;

        ~VulkanTexture();

    private:
        // cube = 6 layers, pixels_per_face laid out +X,-X,+Y,-Y,+Z,-Z
        VulkanTexture(
            const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool, int width, int height,
            const std::vector<std::vector<uint8_t>> &rgba_layers, bool cube);

        void upload(VkCommandPool pool, const std::vector<std::vector<uint8_t>> &rgba_layers);
        void record_layout_transition(
            VkCommandBuffer cmd, VkImageLayout old_layout, VkImageLayout new_layout) const;

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
