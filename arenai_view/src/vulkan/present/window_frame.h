#ifndef ARENAI_VK_WINDOW_FRAME_H
#define ARENAI_VK_WINDOW_FRAME_H

#include <functional>
#include <memory>
#include <vector>

#include "../core/device.h"
#include "../core/render_target.h"
#include "../core/vk.h"
#include "./swap_chain.h"

namespace arenai::view {

    class WindowFrameContext {
    public:
        static constexpr int FRAME_SLOTS = 2;
        static constexpr int WINDOW_MSAA_SAMPLES = 4;

        WindowFrameContext(
            std::shared_ptr<VulkanDevice> device, const VkSurfaceKHR &surface,
            std::function<VkExtent2D()> framebuffer_extent);

        WindowFrameContext(const WindowFrameContext &) = delete;
        WindowFrameContext &operator=(const WindowFrameContext &) = delete;

        bool ensure_frame_begun();
        bool frame_active() const;

        VkCommandBuffer cmd() const;
        int slot() const;

        VkImageView swapchain_view() const;
        VkFormat swapchain_format() const;
        // sample count every pipeline drawing inside a swapchain pass must use
        VkSampleCountFlagBits samples() const;

        int width() const;
        int height() const;

        void begin_swapchain_pass(bool load_existing, bool clear) const;
        void end_swapchain_pass();

        void present();

        void handle_resize();

        ~WindowFrameContext();

    private:
        struct Slot {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkFence in_flight = VK_NULL_HANDLE;
            VkSemaphore image_acquired = VK_NULL_HANDLE;
            bool submitted = false;
        };

        void wait_all_fences();
        bool recreate_swapchain();
        void create_msaa_target();

        std::shared_ptr<VulkanDevice> device_;
        std::unique_ptr<SwapChain> swapchain_;
        bool swapchain_valid_;
        VkSampleCountFlagBits samples_;
        std::unique_ptr<Target> msaa_color_;

        VkCommandPool pool_;
        Slot slots_[FRAME_SLOTS];

        std::vector<VkSemaphore> render_finished_;

        int slot_index_ = 0;
        uint32_t image_index_ = 0;
        bool frame_active_ = false;
        bool image_written_ = false;
    };

}// namespace arenai::view

#endif// ARENAI_VK_WINDOW_FRAME_H
