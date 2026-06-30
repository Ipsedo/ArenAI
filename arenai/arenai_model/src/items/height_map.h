//
// Created by samuel on 23/03/2023.
//

#ifndef ARENAI_HEIGHT_MAP_H
#define ARENAI_HEIGHT_MAP_H

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>

#include <arenai_model/item.h>
#include <arenai_model/shapes.h>
#include <arenai_utils/file_reader.h>

#include "bullet_item.h"

class HeightMapItem final : public BulletItem {
public:
    HeightMapItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &img_reader,
        const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

    ~HeightMapItem() override;

protected:
    glm::vec3 _get_scale() override;

private:
    btHeightfieldTerrainShape *map;
    btRigidBody *body;

    std::string shape_id;
    glm::vec3 scale;

    int map_width;
    int map_height;

    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

    std::vector<float> image_grey;

    float get_height(int x, int z) const;
    glm::vec3 compute_vertex_normal(int x, int z) const;
    glm::vec3 make_pos(int x, int z, int min_height, int max_height) const;

    void
    build_render_mesh(glm::vec3 aabb_min, glm::vec3 aabb_max, float min_height, float max_height);
};

#endif// ARENAI_HEIGHT_MAP_H
