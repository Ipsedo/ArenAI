//
// Created by samuel on 23/03/2023.
//

#ifndef PHYVR_HEIGHT_MAP_H
#define PHYVR_HEIGHT_MAP_H

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>

#include <phyvr_model/item.h>
#include <phyvr_model/shapes.h>
#include <phyvr_utils/file_reader.h>

class HeightMapItem final : public Item, public btTriangleCallback {
public:
    HeightMapItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &img_reader,
        const std::string &height_map_file, glm::vec3 pos, glm::vec3 scale);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

    void processTriangle(btVector3 *triangle, int partid, int triangleindex) override;

    ~HeightMapItem() override;

protected:
    glm::vec3 _get_scale() override;

private:
    std::string shape_id;
    glm::vec3 scale;

    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

    btRigidBody *body;
    btHeightfieldTerrainShape *map;
};

#endif// PHYVR_HEIGHT_MAP_H
