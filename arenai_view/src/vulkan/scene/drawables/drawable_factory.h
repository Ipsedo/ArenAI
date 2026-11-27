#ifndef ARENAI_VK_DRAWABLE_FACTORY_H
#define ARENAI_VK_DRAWABLE_FACTORY_H

#include <memory>
#include <mutex>

#include <arenai_view/drawable.h>

#include "../../core/device.h"
#include "../../core/vk.h"

namespace arenai::view {

    class VulkanDrawableFactory final : public AbstractDrawableFactory {
    public:
        explicit VulkanDrawableFactory(std::shared_ptr<VulkanDevice> device);

        VulkanDrawableFactory(const VulkanDrawableFactory &) = delete;
        VulkanDrawableFactory &operator=(const VulkanDrawableFactory &) = delete;

        std::shared_ptr<AbstractMesh>
        make_mesh(const std::vector<std::tuple<float, float, float>> &vertices) override;

        std::shared_ptr<AbstractCubeTexture> make_cube_texture(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path) override;

        std::unique_ptr<AbstractDrawable>
        make_cube_map(const std::shared_ptr<AbstractCubeTexture> &texture) override;

        std::unique_ptr<AbstractDrawable>
        make_diffuse(const std::shared_ptr<AbstractMesh> &mesh, glm::vec4 color) override;

        std::unique_ptr<AbstractDrawable>
        make_hatched(const std::shared_ptr<AbstractMesh> &mesh, glm::vec4 color) override;

        ~VulkanDrawableFactory() override;

    private:
        std::shared_ptr<VulkanDevice> device_;
        VkCommandPool upload_pool_;
        std::mutex upload_mutex_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DRAWABLE_FACTORY_H
