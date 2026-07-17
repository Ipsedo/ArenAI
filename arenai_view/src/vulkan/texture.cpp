//
// Created by samuel on 17/07/2026.
//

#include "./texture.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "./errors.h"

namespace arenai::view {

    namespace {
        std::vector<uint8_t>
        to_rgba(const int width, const int height, const int channels, const uint8_t *pixels) {
            if (channels != 3 && channels != 4)
                throw std::runtime_error("image need 3 or 4 channels");

            const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
            std::vector<uint8_t> rgba(count * 4);
            if (channels == 4) {
                std::memcpy(rgba.data(), pixels, rgba.size());
            } else {
                for (size_t i = 0; i < count; i++) {
                    rgba[i * 4 + 0] = pixels[i * 3 + 0];
                    rgba[i * 4 + 1] = pixels[i * 3 + 1];
                    rgba[i * 4 + 2] = pixels[i * 3 + 2];
                    rgba[i * 4 + 3] = 255;
                }
            }
            return rgba;
        }
    }// namespace

    VulkanTexture::VulkanTexture(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool, const int width,
        const int height, const int channels, const uint8_t *pixels)
        : VulkanTexture(
              device, pool, width, height, {to_rgba(width, height, channels, pixels)}, false) {}

    std::unique_ptr<VulkanTexture> VulkanTexture::make_white(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool) {
        constexpr uint8_t white[4] = {255, 255, 255, 255};
        return std::make_unique<VulkanTexture>(device, pool, 1, 1, 4, white);
    }

    std::unique_ptr<VulkanTexture> VulkanTexture::from_png(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::filesystem::path &png_path) {
        const auto [width, height, channels, pixels] = file_reader->read_png(png_path);
        return std::make_unique<VulkanTexture>(
            device, pool, width, height, channels, pixels.data());
    }

    std::unique_ptr<VulkanTexture> VulkanTexture::cube_from_pngs(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::filesystem::path &pngs_root_path) {
        // Vulkan cube layer order: +X, -X, +Y, -Y, +Z, -Z
        constexpr const char *faces[] = {"posx.png", "negx.png", "posy.png",
                                         "negy.png", "posz.png", "negz.png"};

        std::vector<std::vector<uint8_t>> layers;
        int width = 0, height = 0;
        for (const auto *face: faces) {
            const auto [face_width, face_height, channels, pixels] =
                file_reader->read_png(pngs_root_path / face);
            if (width == 0) {
                width = face_width;
                height = face_height;
            } else if (face_width != width || face_height != height)
                throw std::runtime_error("cube map faces have mismatched sizes");
            layers.push_back(to_rgba(face_width, face_height, channels, pixels.data()));
        }

        return std::unique_ptr<VulkanTexture>(
            new VulkanTexture(device, pool, width, height, layers, true));
    }

    VulkanTexture::VulkanTexture(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool, const int width,
        const int height, const std::vector<std::vector<uint8_t>> &rgba_layers, const bool cube)
        : device_(device), image_(VK_NULL_HANDLE), allocation_(VK_NULL_HANDLE),
          view_(VK_NULL_HANDLE), sampler_(VK_NULL_HANDLE), width_(width), height_(height),
          layers_(static_cast<uint32_t>(rgba_layers.size())) {
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = layers_;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        vk_check(
            vmaCreateImage(
                device_->allocator(), &image_info, &alloc_info, &image_, &allocation_, nullptr),
            "vmaCreateImage (texture)");

        upload(pool, rgba_layers);

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image_;
        view_info.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers_};
        vk_check(
            vkCreateImageView(device_->handle(), &view_info, nullptr, &view_),
            "vkCreateImageView (texture)");

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(
            vkCreateSampler(device_->handle(), &sampler_info, nullptr, &sampler_),
            "vkCreateSampler (texture)");
    }

    void VulkanTexture::upload(
        const VkCommandPool pool, const std::vector<std::vector<uint8_t>> &rgba_layers) {
        const size_t layer_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 4;

        VkBufferCreateInfo staging_info{};
        staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_info.size = layer_size * layers_;
        staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                   | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_mapped{};
        vk_check(
            vmaCreateBuffer(
                device_->allocator(), &staging_info, &staging_alloc_info, &staging,
                &staging_allocation, &staging_mapped),
            "vmaCreateBuffer (texture staging)");

        auto *dst = static_cast<uint8_t *>(staging_mapped.pMappedData);
        for (size_t layer = 0; layer < rgba_layers.size(); layer++)
            std::memcpy(dst + layer * layer_size, rgba_layers[layer].data(), layer_size);

        device_->immediate_submit(pool, [&](const VkCommandBuffer cmd) {
            record_layout_transition(
                cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            std::vector<VkBufferImageCopy> copies;
            for (uint32_t layer = 0; layer < layers_; layer++) {
                VkBufferImageCopy copy{};
                copy.bufferOffset = layer * layer_size;
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
                copy.imageExtent = {
                    static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};
                copies.push_back(copy);
            }
            vkCmdCopyBufferToImage(
                cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<uint32_t>(copies.size()), copies.data());

            record_layout_transition(
                cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

        vmaDestroyBuffer(device_->allocator(), staging, staging_allocation);
    }

    void VulkanTexture::record_layout_transition(
        const VkCommandBuffer cmd, const VkImageLayout old_layout,
        const VkImageLayout new_layout) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image_;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers_};

        VkPipelineStageFlags src_stage, dst_stage;
        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    VkImageView VulkanTexture::view() const { return view_; }

    VkSampler VulkanTexture::sampler() const { return sampler_; }

    VulkanTexture::~VulkanTexture() {
        vkDestroySampler(device_->handle(), sampler_, nullptr);
        vkDestroyImageView(device_->handle(), view_, nullptr);
        vmaDestroyImage(device_->allocator(), image_, allocation_);
    }

}// namespace arenai::view
