//
// Created by samuel on 17/07/2026.
//

#include "./window_frame.h"

#include <utility>

#include "./errors.h"
#include "./render_target.h"

namespace arenai::view {

    WindowFrameContext::WindowFrameContext(
        std::shared_ptr<VulkanDevice> device, const VkSurfaceKHR surface,
        std::function<VkExtent2D()> framebuffer_extent)
        : device_(std::move(device)),
          swapchain_(std::make_unique<Swapchain>(device_, surface, std::move(framebuffer_extent))),
          swapchain_valid_(swapchain_->handle() != VK_NULL_HANDLE),
          pool_(device_->make_command_pool()) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        for (auto &slot: slots_) {
            vk_check(
                vkAllocateCommandBuffers(device_->handle(), &alloc_info, &slot.cmd),
                "vkAllocateCommandBuffers (window frame)");
            vk_check(
                vkCreateFence(device_->handle(), &fence_info, nullptr, &slot.in_flight),
                "vkCreateFence (window frame)");
            vk_check(
                vkCreateSemaphore(
                    device_->handle(), &semaphore_info, nullptr, &slot.image_acquired),
                "vkCreateSemaphore (acquire)");
        }

        render_finished_.resize(swapchain_->image_count());
        for (auto &semaphore: render_finished_)
            vk_check(
                vkCreateSemaphore(device_->handle(), &semaphore_info, nullptr, &semaphore),
                "vkCreateSemaphore (render finished)");
    }

    bool WindowFrameContext::ensure_frame_begun() {
        if (frame_active_) return true;
        if (!swapchain_valid_) {
            // minimized at the last resize: retry, the window may be back
            device_->wait_idle();
            swapchain_valid_ = swapchain_->recreate();
            if (!swapchain_valid_) return false;
        }

        auto &slot = slots_[slot_index_];
        if (slot.submitted) {
            vk_check(
                vkWaitForFences(device_->handle(), 1, &slot.in_flight, VK_TRUE, UINT64_MAX),
                "vkWaitForFences (window frame)");
            vk_check(
                vkResetFences(device_->handle(), 1, &slot.in_flight),
                "vkResetFences (window frame)");
            slot.submitted = false;
        }

        // acquire, recreating the swapchain when it no longer matches the
        // surface (resize, fullscreen toggle)
        while (true) {
            const VkResult result = swapchain_->acquire(slot.image_acquired, &image_index_);
            if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) break;
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                device_->wait_idle();
                swapchain_valid_ = swapchain_->recreate();
                if (!swapchain_valid_) return false;
                continue;
            }
            vk_check(result, "vkAcquireNextImageKHR");
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(slot.cmd, &begin_info), "vkBeginCommandBuffer (window)");

        // the whole frame is rewritten: discard the previous content
        record_image_barrier(
            slot.cmd, swapchain_->image(image_index_), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        frame_active_ = true;
        image_written_ = false;
        return true;
    }

    bool WindowFrameContext::frame_active() const { return frame_active_; }

    VkCommandBuffer WindowFrameContext::cmd() const { return slots_[slot_index_].cmd; }

    int WindowFrameContext::slot() const { return slot_index_; }

    VkImageView WindowFrameContext::swapchain_view() const {
        return swapchain_->view(image_index_);
    }

    VkFormat WindowFrameContext::swapchain_format() const { return swapchain_->format(); }

    int WindowFrameContext::width() const { return swapchain_->width(); }

    int WindowFrameContext::height() const { return swapchain_->height(); }

    void WindowFrameContext::begin_swapchain_pass(const bool load_existing, const bool clear) {
        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = swapchain_->view(image_index_);
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = load_existing && image_written_ ? VK_ATTACHMENT_LOAD_OP_LOAD
                                  : clear                         ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                                  : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue.color = {{0.f, 0.f, 0.f, 1.f}};

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {
            {0, 0},
            {static_cast<uint32_t>(swapchain_->width()),
             static_cast<uint32_t>(swapchain_->height())}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;

        const VkCommandBuffer command_buffer = cmd();
        if (image_written_) {
            // order this pass after the previous one on the same image
            record_image_barrier(
                command_buffer, swapchain_->image(image_index_), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        }

        vkCmdBeginRendering(command_buffer, &rendering_info);

        // negative height: GL orientation preserved, row 0 = top of frame
        const VkViewport viewport{
            0.f,
            static_cast<float>(swapchain_->height()),
            static_cast<float>(swapchain_->width()),
            -static_cast<float>(swapchain_->height()),
            0.f,
            1.f};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        const VkRect2D scissor{
            {0, 0},
            {static_cast<uint32_t>(swapchain_->width()),
             static_cast<uint32_t>(swapchain_->height())}};
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    }

    void WindowFrameContext::end_swapchain_pass() {
        vkCmdEndRendering(cmd());
        image_written_ = true;
    }

    void WindowFrameContext::present() {
        if (!frame_active_) return;

        auto &slot = slots_[slot_index_];

        record_image_barrier(
            slot.cmd, swapchain_->image(image_index_), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        vk_check(vkEndCommandBuffer(slot.cmd), "vkEndCommandBuffer (window)");

        const VkSemaphore finished = render_finished_[image_index_];
        device_->submit(
            slot.cmd, slot.image_acquired, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, finished,
            slot.in_flight);
        slot.submitted = true;

        const VkResult result = device_->present(swapchain_->handle(), image_index_, finished);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            device_->wait_idle();
            swapchain_valid_ = swapchain_->recreate();
        } else {
            vk_check(result, "vkQueuePresentKHR");
        }

        frame_active_ = false;
        slot_index_ = (slot_index_ + 1) % FRAME_SLOTS;
    }

    void WindowFrameContext::handle_resize() {
        // the resize callback fires between frames (during event polling)
        device_->wait_idle();
        swapchain_valid_ = swapchain_->recreate();

        // render-finished semaphores are per swapchain image
        if (swapchain_valid_ && render_finished_.size() != swapchain_->image_count()) {
            for (const auto semaphore: render_finished_)
                vkDestroySemaphore(device_->handle(), semaphore, nullptr);
            render_finished_.assign(swapchain_->image_count(), VK_NULL_HANDLE);
            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (auto &semaphore: render_finished_)
                vk_check(
                    vkCreateSemaphore(device_->handle(), &semaphore_info, nullptr, &semaphore),
                    "vkCreateSemaphore (render finished)");
        }
    }

    void WindowFrameContext::wait_all_fences() {
        for (auto &slot: slots_)
            if (slot.submitted)
                vkWaitForFences(device_->handle(), 1, &slot.in_flight, VK_TRUE, UINT64_MAX);
    }

    WindowFrameContext::~WindowFrameContext() {
        wait_all_fences();
        device_->wait_idle();
        for (const auto semaphore: render_finished_)
            vkDestroySemaphore(device_->handle(), semaphore, nullptr);
        for (auto &slot: slots_) {
            vkDestroySemaphore(device_->handle(), slot.image_acquired, nullptr);
            vkDestroyFence(device_->handle(), slot.in_flight, nullptr);
        }
        vkDestroyCommandPool(device_->handle(), pool_, nullptr);
        swapchain_.reset();
    }

}// namespace arenai::view
