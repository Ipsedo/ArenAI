//
// Created by samuel on 02/04/2023.
//

#include "./chassis.h"

#include <glm/gtc/type_ptr.hpp>

ChassisItem::ChassisItem(AAssetManager *mgr, glm::vec3 position,
                         glm::vec3 scale, float mass)
    : ConvexItem("tank_chassis",
                 std::make_shared<ObjShape>(mgr, "obj/anubis_chassis.obj"),
                 position, scale, mass) {}

glm::vec3 ChassisItem::get_pos_() {
  btTransform chassis_tr = ConvexItem::get_body()->getWorldTransform();
  btVector3 tmp_pos = chassis_tr.getOrigin();

  glm::vec3 chassis_pos(tmp_pos.x(), tmp_pos.y(), tmp_pos.z());

  return chassis_pos;
}

glm::vec3 ChassisItem::pos() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 7, -20, 1);
}

glm::vec3 ChassisItem::look() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 0, 1, 1);
}

glm::vec3 ChassisItem::up() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 1, 0, 0);
}
