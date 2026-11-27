#include "./drawable_factory.h"

#include <stdexcept>
#include <utility>

#include "../../core/texture.h"
#include "./cube_map.h"
#include "./cube_texture.h"
#include "./diffuse.h"
#include "./mesh.h"

namespace arenai::view {

    namespace {
        template<typename Vulkan, typename Abstract>
        std::shared_ptr<const Vulkan> as_vulkan(const std::shared_ptr<Abstract> &resource) {
            auto vulkan = std::dynamic_pointer_cast<const Vulkan>(resource);
            if (!vulkan) throw std::invalid_argument("resource made by another drawable factory");
            return vulkan;
        }
    }// namespace

    VulkanDrawableFactory::VulkanDrawableFactory(std::shared_ptr<VulkanDevice> device)
        : device_(std::move(device)), upload_pool_(device_->make_command_pool()) {}

    std::shared_ptr<AbstractMesh>
    VulkanDrawableFactory::make_mesh(const std::vector<std::tuple<float, float, float>> &vertices) {
        std::lock_guard lock(upload_mutex_);
        return std::make_shared<VulkanMesh>(device_, upload_pool_, vertices);
    }

    std::shared_ptr<AbstractCubeTexture> VulkanDrawableFactory::make_cube_texture(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::filesystem::path &pngs_root_path) {
        std::lock_guard lock(upload_mutex_);
        return std::make_shared<VulkanCubeTexture>(
            VulkanTexture::cube_from_pngs(device_, upload_pool_, file_reader, pngs_root_path));
    }

    std::unique_ptr<AbstractDrawable>
    VulkanDrawableFactory::make_cube_map(const std::shared_ptr<AbstractCubeTexture> &texture) {
        return std::make_unique<VulkanCubeMap>(as_vulkan<VulkanCubeTexture>(texture));
    }

    std::unique_ptr<AbstractDrawable> VulkanDrawableFactory::make_diffuse(
        const std::shared_ptr<AbstractMesh> &mesh, const glm::vec4 color) {
        return std::make_unique<VulkanDiffuse>(as_vulkan<VulkanMesh>(mesh), color);
    }

    VulkanDrawableFactory::~VulkanDrawableFactory() {
        vkDestroyCommandPool(device_->handle(), upload_pool_, nullptr);
    }

}// namespace arenai::view
