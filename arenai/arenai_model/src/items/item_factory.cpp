//
// Created by claude on 30/06/2026.
//

#include <arenai_model/item_factory.h>

#include "convex.h"
#include "height_map.h"

std::shared_ptr<Item> make_sphere_item(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass) {
    return std::make_shared<SphereItem>(std::move(name), file_reader, position, scale, mass);
}

std::shared_ptr<Item> make_cube_item(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass) {
    return std::make_shared<CubeItem>(std::move(name), file_reader, position, scale, mass);
}

std::shared_ptr<Item> make_tetra_item(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass) {
    return std::make_shared<TetraItem>(std::move(name), file_reader, position, scale, mass);
}

std::shared_ptr<Item> make_cylinder_item(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass) {
    return std::make_shared<CylinderItem>(std::move(name), file_reader, position, scale, mass);
}

std::shared_ptr<Item> make_height_map_item(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
    const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale) {
    return std::make_shared<HeightMapItem>(
        std::move(name), file_reader, height_map_file, pos, scale);
}
