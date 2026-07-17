//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_WINDOW_FRAME_H
#define ARENAI_VK_WINDOW_FRAME_H

#include <memory>
#include <vector>

#include "./device.h"
#include "./swapchain.h"
#include "./vk.h"

namespace arenai::view {

    // Orchestrates the windowed frame: 2 frames in flight over the swapchain,
    // shared by the backend (UI passes, present), the player renderer (scene
    // + composite + HUD) and the Rml render interface. One frame spans the
    // whole application call sequence — player draw, then optionally the UI
    // overlay, then present() — so the swapchain image is acquired lazily on
    // the first draw and the command buffer stays open until present().
    class WindowFrameContext {
    public:
        static constexpr int FRAME_SLOTS = 2;

        WindowFrameContext(std::shared_ptr<VulkanDevice> device, VkSurfaceKHR surface);

        WindowFrameContext(const WindowFrameContext &) = delete;
        WindowFrameContext &operator=(const WindowFrameContext &) = delete;

        // waits the slot fence, acquires a swapchain image and begins the
        // command buffer; idempotent within a frame. Returns false when the
        // window is minimized (0x0): the frame is skipped entirely.
        bool ensure_frame_begun();
        bool frame_active() const;

        VkCommandBuffer cmd() const;
        int slot() const;
        VkImageView swapchain_view() const;
        VkFormat swapchain_format() const;
        int width() const;
        int height() const;

        // rendering scope on the swapchain image: load_existing keeps the
        // pixels already recorded this frame (UI overlay over the game),
        // otherwise the image is cleared to black (UI-only frame) or left
        // undefined (clear = false: the composite pass covers every pixel)
        void begin_swapchain_pass(bool load_existing, bool clear);
        void end_swapchain_pass();

        // barrier to PRESENT_SRC, submit, present, advance the slot; no-op
        // when no frame was begun (minimized window)
        void present();

        // called from the resize callback: waits the device idle and
        // recreates the swapchain at the new surface size
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

        std::shared_ptr<VulkanDevice> device_;
        std::unique_ptr<Swapchain> swapchain_;
        bool swapchain_valid_;

        VkCommandPool pool_;
        Slot slots_[FRAME_SLOTS];
        // one per swapchain image: signaled by the submit, waited by present
        std::vector<VkSemaphore> render_finished_;

        int slot_index_ = 0;
        uint32_t image_index_ = 0;
        bool frame_active_ = false;
        bool image_written_ = false;
    };

}// namespace arenai::view

#endif// ARENAI_VK_WINDOW_FRAME_H
