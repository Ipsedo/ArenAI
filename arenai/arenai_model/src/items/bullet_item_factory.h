//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_BULLET_ITEM_FACTORY_H
#define ARENAI_BULLET_ITEM_FACTORY_H

#include <arenai_model/item_factory.h>

namespace arenai::model {

    class BulletPhysicEngine;

    class BulletItemFactory final : public ItemFactory {
    public:
        explicit BulletItemFactory(BulletPhysicEngine &engine);

        std::shared_ptr<Item> make_sphere_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) override;

        std::shared_ptr<Item> make_cube_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) override;

        std::shared_ptr<Item> make_tetra_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) override;

        std::shared_ptr<Item> make_cylinder_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) override;

        std::shared_ptr<Item> make_height_map_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale) override;

    private:
        BulletPhysicEngine &engine;
    };

}// namespace arenai::model

#endif// ARENAI_BULLET_ITEM_FACTORY_H
