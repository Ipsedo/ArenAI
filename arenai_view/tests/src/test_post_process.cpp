//
// Created by samuel on 15/07/2026.
//

#include <cmath>
#include <numeric>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include "vulkan/core/buffer.h"
#include "vulkan/core/descriptors.h"
#include "vulkan/core/render_target.h"
#include "vulkan/post/post_process.h"
#include "vulkan/vulkan_backend.h"

using namespace arenai;
using namespace arenai::view;

namespace {

    constexpr int WIDTH = 64, HEIGHT = 64;

    // smoke-test harness of the whole player post-processing pipeline on a
    // headless device: every effect shader (SSAO, blurs, bloom, god rays,
    // composite) is compiled and drawn once, the composited frame is read
    // back from an offscreen target standing in for the swapchain image
    class PostProcessHarness {
    public:
        PostProcessHarness()
            : backend_(std::make_unique<VulkanBackend>()),
              device_(std::dynamic_pointer_cast<VulkanRenderContext>(backend_->render_context())
                          ->device()),
              descriptors_(device_), pool_(device_->make_command_pool()),
              post_process_(
                  device_, &descriptors_, WIDTH, HEIGHT,
                  make_default_post_processing_effects(device_, &descriptors_, WIDTH, HEIGHT)) {}

        VulkanPostProcess &post_process() { return post_process_; }

        // records one full frame and returns the sum of the output pixels
        long run_frame(const int width, const int height) {
            const glm::mat4 proj_matrix = glm::perspectiveRH_ZO(
                glm::quarter_pi<float>(), static_cast<float>(width) / static_cast<float>(height),
                1.f, 2000.f * std::sqrt(3.f));
            // sun in front of the camera, so the god-rays pass runs too
            const glm::vec3 sun_dir_view = glm::normalize(glm::vec3(0.1f, 0.3f, -1.f));

            const Target output(
                device_, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
            const HostVisibleBuffer readback(
                device_, static_cast<size_t>(width) * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

            device_->immediate_submit(pool_, [&](const VkCommandBuffer cmd) {
                // scene pass: the red clear alone feeds the effect chain
                post_process_.begin_scene_pass(cmd);
                post_process_.run_effects(cmd, proj_matrix, sun_dir_view);

                // composite into the output target, as the player renderer
                // does into the swapchain image
                record_image_barrier(
                    cmd, output.image(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

                VkRenderingAttachmentInfo color_attachment{};
                color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                color_attachment.imageView = output.view();
                color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo rendering_info{};
                rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering_info.renderArea = {
                    {0, 0}, {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}};
                rendering_info.layerCount = 1;
                rendering_info.colorAttachmentCount = 1;
                rendering_info.pColorAttachments = &color_attachment;
                vkCmdBeginRendering(cmd, &rendering_info);

                const VkRect2D scissor{
                    {0, 0}, {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}};
                vkCmdSetScissor(cmd, 0, 1, &scissor);

                post_process_.composite_within(cmd, output.format(), width, height);

                vkCmdEndRendering(cmd);

                record_image_barrier(
                    cmd, output.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
                vkCmdCopyImageToBuffer(
                    cmd, output.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1,
                    &copy);
            });

            readback.invalidate();
            const auto *pixels = static_cast<const uint8_t *>(readback.data());
            return std::accumulate(
                pixels, pixels + static_cast<size_t>(width) * height * 4, 0L,
                [](const long acc, const uint8_t p) { return acc + static_cast<long>(p); });
        }

        ~PostProcessHarness() {
            device_->wait_idle();
            vkDestroyCommandPool(device_->handle(), pool_, nullptr);
        }

    private:
        std::unique_ptr<VulkanBackend> backend_;
        std::shared_ptr<VulkanDevice> device_;
        DescriptorAllocator descriptors_;
        VkCommandPool pool_;
        VulkanPostProcess post_process_;
    };

}// namespace

TEST(TestPostProcess, FullPipelineRuns) {
    PostProcessHarness harness;

    // the cleared scene must reach the output through the composite pass
    ASSERT_GT(harness.run_frame(WIDTH, HEIGHT), 0L);
}

TEST(TestPostProcess, SurvivesResize) {
    PostProcessHarness harness;
    ASSERT_GT(harness.run_frame(WIDTH, HEIGHT), 0L);

    constexpr int NEW_WIDTH = 32, NEW_HEIGHT = 48;
    harness.post_process().resize(NEW_WIDTH, NEW_HEIGHT);
    ASSERT_GT(harness.run_frame(NEW_WIDTH, NEW_HEIGHT), 0L);
}
