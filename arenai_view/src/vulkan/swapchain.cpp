//
// Created by samuel on 17/07/2026.
//

#include "./swapchain.h"

#include <algorithm>
#include <utility>

#include "./errors.h"

namespace arenai::view {

    Swapchain::Swapchain(
        std::shared_ptr<VulkanDevice> device, const VkSurfaceKHR surface,
        std::function<VkExtent2D()> framebuffer_extent)
        : device_(std::move(device)), surface_(surface),
          framebuffer_extent_(std::move(framebuffer_extent)), swapchain_(VK_NULL_HANDLE),
          format_(VK_FORMAT_UNDEFINED), extent_{0, 0} {
        recreate();
    }

    bool Swapchain::matches_framebuffer() const {
        const auto [width, height] = framebuffer_extent_();
        if (width == 0 || height == 0) return true;
        return width == extent_.width && height == extent_.height;
    }

    bool Swapchain::recreate() {
        VkSurfaceCapabilitiesKHR capabilities;
        vk_check(
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->physical(), surface_, &capabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        VkExtent2D extent = capabilities.currentExtent;
        if (extent.width == UINT32_MAX) {
            // the surface has no fixed extent (Wayland): the swapchain sets
            // the size — use the framebuffer size, clamped to the surface caps
            extent = framebuffer_extent_();
            if (extent.width == 0 || extent.height == 0) return false;
            extent.width = std::clamp(
                extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std::clamp(
                extent.height, capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);
        }
        if (extent.width == 0 || extent.height == 0) return false;

        // no-sRGB format: the shaders encode manually, like the GL pipeline
        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_->physical(), surface_, &format_count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            device_->physical(), surface_, &format_count, formats.data());

        VkSurfaceFormatKHR chosen = formats.front();
        for (const auto &candidate: formats)
            if (candidate.format == VK_FORMAT_B8G8R8A8_UNORM
                || candidate.format == VK_FORMAT_R8G8B8A8_UNORM) {
                chosen = candidate;
                break;
            }

        uint32_t image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0)
            image_count = std::min(image_count, capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = chosen.format;
        create_info.imageColorSpace = chosen.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform = capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        // FIFO: vsync pacing, guaranteed available (matches eglSwapBuffers)
        create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = swapchain_;

        VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
        vk_check(
            vkCreateSwapchainKHR(device_->handle(), &create_info, nullptr, &new_swapchain),
            "vkCreateSwapchainKHR");

        destroy_views();
        if (swapchain_ != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device_->handle(), swapchain_, nullptr);
        swapchain_ = new_swapchain;
        format_ = chosen.format;
        extent_ = extent;

        uint32_t count = 0;
        vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &count, nullptr);
        images_.resize(count);
        vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &count, images_.data());

        views_.resize(count);
        for (uint32_t i = 0; i < count; i++) {
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = images_[i];
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = format_;
            view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vk_check(
                vkCreateImageView(device_->handle(), &view_info, nullptr, &views_[i]),
                "vkCreateImageView (swapchain)");
        }
        return true;
    }

    VkResult Swapchain::acquire(const VkSemaphore signal, uint32_t *image_index) const {
        return vkAcquireNextImageKHR(
            device_->handle(), swapchain_, UINT64_MAX, signal, VK_NULL_HANDLE, image_index);
    }

    VkSwapchainKHR Swapchain::handle() const { return swapchain_; }

    VkFormat Swapchain::format() const { return format_; }

    int Swapchain::width() const { return static_cast<int>(extent_.width); }

    int Swapchain::height() const { return static_cast<int>(extent_.height); }

    uint32_t Swapchain::image_count() const { return static_cast<uint32_t>(images_.size()); }

    VkImage Swapchain::image(const uint32_t index) const { return images_[index]; }

    VkImageView Swapchain::view(const uint32_t index) const { return views_[index]; }

    void Swapchain::destroy_views() {
        for (const auto view: views_) vkDestroyImageView(device_->handle(), view, nullptr);
        views_.clear();
    }

    Swapchain::~Swapchain() {
        destroy_views();
        if (swapchain_ != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device_->handle(), swapchain_, nullptr);
    }

}// namespace arenai::view
