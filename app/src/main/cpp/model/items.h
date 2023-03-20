//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_ITEMS_H
#define PHYVR_ITEMS_H

#include <memory>
#include <string>

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>

#include "shapes.h"

class Item {
public:
    Item(std::string name);

    virtual std::shared_ptr<Shape> get_shape() = 0;

    std::string get_name();

    glm::mat4 get_model_matrix();

    virtual btRigidBody *get_body() = 0;

private:
    std::string name;
protected:
    virtual glm::vec3 _get_scale() = 0;
};

class ConvexItem : public Item {
public:
    ConvexItem(std::string name, const std::shared_ptr<Shape> &shape, glm::vec3 position,
               glm::vec3 scale, float mass);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

protected:
    glm::vec3 _get_scale() override;

private:
private:
    std::string name;

    std::shared_ptr<Shape> shape;

    btRigidBody *body;
    btCollisionShape *collision_shape;

    glm::vec3 scale;
};

class HeightMapItem : public Item, public btTriangleCallback {
public:
    HeightMapItem(std::string name, AAssetManager *mgr,
                  const std::string &height_map_file, glm::vec3 pos,
                  glm::vec3 scale);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

    void processTriangle(btVector3 *triangle, int partid, int triangleindex) override;

protected:
    glm::vec3 _get_scale() override;

private:
    std::shared_ptr<Shape> shape;
    glm::vec3 scale;

    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

    btRigidBody *body;
};

#endif //PHYVR_ITEMS_H
