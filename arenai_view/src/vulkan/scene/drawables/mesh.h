#ifndef ARENAI_VK_MESH_H
#define ARENAI_VK_MESH_H

#include <memory>
#include <tuple>
#include <vector>

#include <arenai_view/drawable.h>

#include "../../core/buffer.h"
#include "../../core/device.h"
#include "../../core/vk.h"

namespace arenai::view {

    class VulkanMesh final : public AbstractMesh {
    public:
        VulkanMesh(
            const std::shared_ptr<VulkanDevice> &device, VkCommandPool pool,
            const std::vector<std::tuple<float, float, float>> &vertices);

        void record_draw(VkCommandBuffer cmd) const;

    private:
        std::unique_ptr<VulkanBuffer> vertices_;
        std::unique_ptr<VulkanBuffer> indices_;
        uint32_t nb_indices_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_MESH_H
