//
// Created by samuel on 23/03/2023.
//

#ifndef PHYVR_CONVEX_H
#define PHYVR_CONVEX_H

#include <phyvr_model/item.h>
#include <phyvr_utils/file_reader.h>

class ConvexItem : public Item {
public:
    ConvexItem(
        std::string name, const std::shared_ptr<Shape> &shape, glm::vec3 position, glm::vec3 scale,
        float mass);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

protected:
    glm::vec3 _get_scale() override;

private:
private:
    std::string name;

    std::shared_ptr<Shape> shape;

    btRigidBody *body;
    btConvexHullShape *collision_shape;

    glm::vec3 scale;
};

class SphereItem : public ConvexItem {
public:
    SphereItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class CubeItem : public ConvexItem {
public:
    CubeItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class CylinderItem : public ConvexItem {
public:
    CylinderItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class TetraItem : public ConvexItem {
public:
    TetraItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

#endif// PHYVR_CONVEX_H
