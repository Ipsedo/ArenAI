//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_DRAWABLE_FACTORY_H
#define ARENAI_VK_DRAWABLE_FACTORY_H

#include <arenai_view/drawable.h>

namespace arenai::view {

    class VulkanDrawableFactory final : public AbstractDrawableFactory {
    public:
        std::unique_ptr<AbstractDrawable> make_cube_map(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path) override;

        std::unique_ptr<AbstractDrawable> make_diffuse(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices, glm::vec4 color) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DRAWABLE_FACTORY_H
