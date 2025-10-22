//
// Created by samuel on 23/03/2023.
//

#ifndef ARENAI_CONVEX_H
#define ARENAI_CONVEX_H

#include <arenai_model/item.h>
#include <arenai_utils/file_reader.h>

class ConvexItem : public Item {
public:
    ConvexItem(
        std::string name, const std::shared_ptr<Shape> &shape, glm::vec3 position, glm::vec3 scale,
        float mass);

    std::shared_ptr<Shape> get_shape() override;

    btRigidBody *get_body() override;

    ~ConvexItem() override;

protected:
    glm::vec3 _get_scale() override;

private:
    std::string name;

    std::shared_ptr<Shape> shape;

    btRigidBody *body;
    btConvexHullShape *collision_shape;

    glm::vec3 scale;
};

class SphereItem final : public ConvexItem {
public:
    SphereItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class CubeItem final : public ConvexItem {
public:
    CubeItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class CylinderItem final : public ConvexItem {
public:
    CylinderItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

class TetraItem final : public ConvexItem {
public:
    TetraItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

#endif// ARENAI_CONVEX_H
