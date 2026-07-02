//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_ITEM_FACTORY_H
#define ARENAI_ITEM_FACTORY_H

#include <filesystem>
#include <memory>
#include <string>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "./item.h"

namespace arenai::model {

    class ItemFactory {
    public:
        virtual ~ItemFactory() = default;

        virtual std::shared_ptr<Item> make_sphere_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) = 0;

        virtual std::shared_ptr<Item> make_cube_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) = 0;

        virtual std::shared_ptr<Item> make_tetra_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) = 0;

        virtual std::shared_ptr<Item> make_cylinder_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass) = 0;

        virtual std::shared_ptr<Item> make_height_map_item(
            std::string name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale) = 0;
    };

}// namespace arenai::model

#endif// ARENAI_ITEM_FACTORY_H
