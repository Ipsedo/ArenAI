//
// Created by samuel on 23/03/2023.
//

#include <utility>

#include <phyvr_model/convex.h>
#include <phyvr_utils/cache.h>
#include <phyvr_utils/singleton.h>

ConvexItem::ConvexItem(
    std::string name, const std::shared_ptr<Shape> &shape, glm::vec3 position, glm::vec3 scale,
    float mass)
    : Item(std::move(name)), shape(shape), scale(scale) {

    auto collision_shape_cache = Singleton<Cache<btCollisionShape *>>::get_singleton();
    auto local_inertia_cache = Singleton<Cache<btVector3>>::get_singleton();

    btVector3 local_inertia(0, 0, 0);

    auto tmp = shape->get_id();

    if (collision_shape_cache->exists(shape->get_id())) {
        collision_shape = collision_shape_cache->get(shape->get_id());
        local_inertia = local_inertia_cache->get(shape->get_id());
    } else {
        auto *convex_hull_shape = new btConvexHullShape();

        for (auto [x, y, z]: shape->get_vertices()) convex_hull_shape->addPoint(btVector3(x, y, z));

        collision_shape = convex_hull_shape;

        collision_shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

        if (mass != 0.f) { collision_shape->calculateLocalInertia(mass, local_inertia); }

        collision_shape_cache->add(shape->get_id(), collision_shape);
        local_inertia_cache->add(shape->get_id(), local_inertia);
    }

    btTransform original_tr;
    original_tr.setIdentity();
    original_tr.setOrigin(btVector3(position.x, position.y, position.z));

    auto *motion_state = new btDefaultMotionState(original_tr);

    btRigidBody::btRigidBodyConstructionInfo body_info(
        mass, motion_state, collision_shape, local_inertia);

    body = new btRigidBody(body_info);
}

std::shared_ptr<Shape> ConvexItem::get_shape() { return shape; }

btRigidBody *ConvexItem::get_body() { return body; }

glm::vec3 ConvexItem::_get_scale() { return scale; }

/*
 * Basic shapes
 */

CubeItem::CubeItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass)
    : ConvexItem(
        std::move(name), std::make_shared<ObjShape>(file_reader, "obj/cube.obj"), position, scale,
        mass) {}

SphereItem::SphereItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass)
    : ConvexItem(
        std::move(name), std::make_shared<ObjShape>(file_reader, "obj/sphere.obj"), position, scale,
        mass) {}

CylinderItem::CylinderItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass)
    : ConvexItem(
        std::move(name), std::make_shared<ObjShape>(file_reader, "obj/cylinder.obj"), position,
        scale, mass) {}

TetraItem::TetraItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 position,
    glm::vec3 scale, float mass)
    : ConvexItem(
        std::move(name), std::make_shared<ObjShape>(file_reader, "obj/tetra.obj"), position, scale,
        mass) {}
