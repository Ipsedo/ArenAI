//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SWAPCHAIN_H
#define ARENAI_VK_SWAPCHAIN_H

#include <functional>
#include <memory>
#include <vector>

#include "../core/device.h"
#include "../core/vk.h"

namespace arenai::view {

    class SwapChain {
    public:
        SwapChain(
            std::shared_ptr<VulkanDevice> device, const VkSurfaceKHR &surface,
            std::function<VkExtent2D()> framebuffer_extent);

        SwapChain(const SwapChain &) = delete;
        SwapChain &operator=(const SwapChain &) = delete;

        bool recreate();

        bool matches_framebuffer() const;

        VkResult acquire(const VkSemaphore &signal, uint32_t *image_index) const;

        VkSwapchainKHR handle() const;
        VkFormat format() const;
        int width() const;
        int height() const;
        uint32_t image_count() const;
        VkImage image(uint32_t index) const;
        VkImageView view(uint32_t index) const;

        ~SwapChain();

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
