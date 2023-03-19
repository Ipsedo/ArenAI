//
// Created by samuel on 19/03/2023.
//

#include "items.h"

#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <btBulletCollisionCommon.h>
#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>

#include "../utils/asset.h"

Item::Item(std::string name) : name(std::move(name)) {

}

std::string Item::get_name() {
    return name;
}

glm::mat4 Item::get_model_matrix() {
    btScalar tmp[16];
    btTransform tr;

    get_body()->getMotionState()->getWorldTransform(tr);

    tr.getOpenGLMatrix(tmp);

    return glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), _get_scale());
}

// ConvexItem / ConvexHull

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

    btRigidBody::btRigidBodyConstructionInfo body_info(mass, motion_state, collision_shape, local_inertia);

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

// Height Map

HeightMapItem::HeightMapItem(std::string name, AAssetManager *mgr,
                             const std::string &height_map_file, glm::vec3 pos,
                             glm::vec3 scale) :
        Item(std::move(name)), scale(scale) {
    libpng_image tmp = read_png(mgr, "heightmap/heightmap6.png");
    img_grey img = to_img_grey(tmp);

    auto map = new btHeightfieldTerrainShape(img.width, img.height, img.pixels, 1.f, 0.f, 1.f,
                                             1, PHY_FLOAT, false);
    map->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
    map->processAllTriangles(
            this,
            btVector3(-2000., -2000., -2000.),
            btVector3(2000., 2000., 2000.));

    btTransform myTransform;
    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

    btVector3 intertie(0.f, 0.f, 0.f);
    auto *motionState = new btDefaultMotionState(myTransform);
    btRigidBody::btRigidBodyConstructionInfo info(0.f, motionState, map, intertie);

    body = new btRigidBody(info);
}

std::shared_ptr<Shape> HeightMapItem::get_shape() {
    return std::make_shared<FromMeshShape>(vertices, normals);
}

btRigidBody *HeightMapItem::get_body() {
    return body;
}

glm::vec3 HeightMapItem::_get_scale() {
    return scale;
}

void HeightMapItem::processTriangle(btVector3 *triangle, int partid, int triangleindex) {
    glm::vec3 p1 = glm::vec3(triangle[0].getX(), triangle[0].getY(), triangle[0].getZ());
    glm::vec3 p2 = glm::vec3(triangle[1].getX(), triangle[1].getY(), triangle[1].getZ());
    glm::vec3 p3 = glm::vec3(triangle[2].getX(), triangle[2].getY(), triangle[2].getZ());

    glm::vec3 n1 = glm::cross(p1 - p2, p3 - p2);
    glm::vec3 n2 = glm::cross(p3 - p2, p1 - p2);

    // y : up axis
    glm::vec3 n = n1.y > 0 ? n1 : n2;

    vertices.emplace_back(p1.x, p1.y, p1.z);
    normals.emplace_back(n.x, n.y, n.z);

    vertices.emplace_back(p2.x, p2.y, p2.z);
    normals.emplace_back(n.x, n.y, n.z);

    vertices.emplace_back(p3.x, p3.y, p3.z);
    normals.emplace_back(n.x, n.y, n.z);
}
