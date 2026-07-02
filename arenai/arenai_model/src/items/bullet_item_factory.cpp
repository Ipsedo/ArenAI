//
// Created by samuel on 30/06/2026.
//

#include "./bullet_item_factory.h"

#include "../bullet_engine.h"
#include "./convex.h"
#include "./height_map.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    BulletItemFactory::BulletItemFactory(BulletPhysicEngine &engine) : engine(engine) {}

    std::shared_ptr<Item> BulletItemFactory::make_sphere_item(
        std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass) {
        auto item =
            std::make_shared<SphereItem>(std::move(name), file_reader, position, scale, mass);
        engine.add_bullet_item(item);
        return item;
    }

    std::shared_ptr<Item> BulletItemFactory::make_cube_item(
        std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass) {
        auto item = std::make_shared<CubeItem>(std::move(name), file_reader, position, scale, mass);
        engine.add_bullet_item(item);
        return item;
    }

    std::shared_ptr<Item> BulletItemFactory::make_tetra_item(
        std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass) {
        auto item =
            std::make_shared<TetraItem>(std::move(name), file_reader, position, scale, mass);
        engine.add_bullet_item(item);
        return item;
    }

    std::shared_ptr<Item> BulletItemFactory::make_cylinder_item(
        std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass) {
        auto item =
            std::make_shared<CylinderItem>(std::move(name), file_reader, position, scale, mass);
        engine.add_bullet_item(item);
        return item;
    }

    std::shared_ptr<Item> BulletItemFactory::make_height_map_item(
        std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale) {
        auto item = std::make_shared<HeightMapItem>(
            std::move(name), file_reader, height_map_file, pos, scale);
        engine.add_bullet_item(item);
        return item;
    }

}// namespace arenai::model
