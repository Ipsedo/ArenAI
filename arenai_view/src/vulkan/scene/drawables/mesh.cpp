#include "./mesh.h"

#include <array>
#include <bit>
#include <cstdint>
#include <unordered_map>

namespace arenai::view {

    namespace {
        struct PositionBits {
            std::array<uint32_t, 3> bits;
            bool operator==(const PositionBits &) const = default;
        };

        struct PositionBitsHash {
            size_t operator()(const PositionBits &position) const {
                size_t seed = 0;
                for (const uint32_t bits: position.bits)
                    seed ^= std::hash<uint32_t>{}(bits) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        struct IndexedPositions {
            std::vector<float> positions;
            std::vector<uint32_t> indices;
        };

        // identical positions share one vertex: flat shading needs no per-face attribute
        IndexedPositions
        index_positions(const std::vector<std::tuple<float, float, float>> &vertices) {
            IndexedPositions indexed;
            indexed.indices.reserve(vertices.size());

            std::unordered_map<PositionBits, uint32_t, PositionBitsHash> index_of;
            index_of.reserve(vertices.size());

            for (const auto &[x, y, z]: vertices) {
                const PositionBits key{
                    {std::bit_cast<uint32_t>(x), std::bit_cast<uint32_t>(y),
                     std::bit_cast<uint32_t>(z)}};
                const auto [entry, inserted] =
                    index_of.try_emplace(key, static_cast<uint32_t>(indexed.positions.size() / 3));
                if (inserted) indexed.positions.insert(indexed.positions.end(), {x, y, z});
                indexed.indices.push_back(entry->second);
            }
            return indexed;
        }
    }// namespace

    VulkanMesh::VulkanMesh(
        const std::shared_ptr<VulkanDevice> &device, const VkCommandPool pool,
        const std::vector<std::tuple<float, float, float>> &vertices)
        : nb_indices_(static_cast<uint32_t>(vertices.size())) {
        const auto [positions, indices] = index_positions(vertices);

        vertices_ = std::make_unique<VulkanBuffer>(
            device, pool, positions.data(), positions.size() * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        indices_ = std::make_unique<VulkanBuffer>(
            device, pool, indices.data(), indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    void VulkanMesh::record_draw(const VkCommandBuffer cmd) const {
        const VkBuffer vertex_buffer = vertices_->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, indices_->handle(), 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cmd, nb_indices_, 1, 0, 0, 0);
    }

}// namespace arenai::view
