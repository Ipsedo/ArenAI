//
// Created by samuel on 23/03/2023.
//

#ifndef ARENAI_HEIGHT_MAP_H
#define ARENAI_HEIGHT_MAP_H

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>

#include <arenai_model/item.h>
#include <arenai_model/shapes.h>
#include <arenai_utils/file_reader.h>

#include "../jolt_item.h"

namespace arenai::model {

    class HeightMapItem final : public JoltItem {
    public:
        HeightMapItem(
            std::string name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &img_reader,
            const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale);

        std::shared_ptr<Shape> get_shape() override;

        JPH::Body *get_body() override;

        ~HeightMapItem() override;

    protected:
        glm::vec3 _get_scale() override;

    private:
        JPH::Body *body;

        std::string shape_id;
        glm::vec3 scale;

        int map_width;
        int map_height;

        std::vector<std::tuple<float, float, float>> vertices;
        std::vector<std::tuple<float, float, float>> normals;

        std::vector<float> image_grey;

        float get_height(int x, int z) const;
        glm::vec3 compute_vertex_normal(int x, int z) const;
        glm::vec3 make_pos(int x, int z, float min_height, float max_height) const;

        void build_render_mesh(
            glm::vec3 aabb_min, glm::vec3 aabb_max, float min_height, float max_height);
    };

}// namespace arenai::model

#endif// ARENAI_HEIGHT_MAP_H
