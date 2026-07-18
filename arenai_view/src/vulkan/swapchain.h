//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SWAPCHAIN_H
#define ARENAI_VK_SWAPCHAIN_H

#include <functional>
#include <memory>
#include <vector>

#include "./device.h"
#include "./vk.h"

namespace arenai::view {

    // FIFO (vsync) swapchain over the window surface, UNORM format — the
    // shaders do their own sRGB encode, exactly like the GL pipeline.
    class Swapchain {
    public:
        // framebuffer_extent: current window framebuffer size, used when the
        // surface reports no fixed extent (Wayland: currentExtent = 0xFFFFFFFF)
        Swapchain(
            std::shared_ptr<VulkanDevice> device, VkSurfaceKHR surface,
            std::function<VkExtent2D()> framebuffer_extent);

        Swapchain(const Swapchain &) = delete;
        Swapchain &operator=(const Swapchain &) = delete;

        // (re)creates the swapchain at the surface's current size; the caller
        // has waited the device idle first. Returns false when the surface is
        // 0x0 (minimized): no swapchain exists until the next recreate.
        bool recreate();

        VkResult acquire(VkSemaphore signal, uint32_t *image_index) const;

        VkSwapchainKHR handle() const;
        VkFormat format() const;
        int width() const;
        int height() const;
        uint32_t image_count() const;
        VkImage image(uint32_t index) const;
        VkImageView view(uint32_t index) const;

        ~Swapchain();

    private:
        void destroy_views();

        std::shared_ptr<VulkanDevice> device_;
        VkSurfaceKHR surface_;
        std::function<VkExtent2D()> framebuffer_extent_;
        VkSwapchainKHR swapchain_;
        VkFormat format_;
        VkExtent2D extent_;
        std::vector<VkImage> images_;
        std::vector<VkImageView> views_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SWAPCHAIN_H
