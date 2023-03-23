//
// Created by samuel on 23/03/2023.
//

#include "./convex.h"

ConvexItem::ConvexItem(std::string name, const std::shared_ptr<Shape> &shape,
                       glm::vec3 position, glm::vec3 scale, float mass) :
        Item(std::move(name)), shape(shape), scale(scale) {
    auto *convex_hull_shape = new btConvexHullShape();

    for (auto [x, y, z]: shape->get_vertices())
        convex_hull_shape->addPoint(btVector3(x, y, z));

    collision_shape = convex_hull_shape;

    collision_shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

    btVector3 local_inertia(0, 0, 0);
    if (mass != 0.f)
        collision_shape->calculateLocalInertia(mass, local_inertia);

    btTransform original_tr;
    original_tr.setIdentity();
    original_tr.setOrigin(btVector3(position.x, position.y, position.z));

    auto *motion_state = new btDefaultMotionState(original_tr);

    btRigidBody::btRigidBodyConstructionInfo body_info(mass, motion_state, collision_shape,
                                                       local_inertia);

    body = new btRigidBody(body_info);
}

std::shared_ptr<Shape> ConvexItem::get_shape() {
    return shape;
}

btRigidBody *ConvexItem::get_body() {
    return body;
}

glm::vec3 ConvexItem::_get_scale() {
    return scale;
}