//
// Created by samuel on 02/04/2023.
//

#include "./canon.h"
#include "../../utils/logging.h"

#include <glm/gtc/type_ptr.hpp>

CanonItem::CanonItem(std::string name, AAssetManager *mgr, glm::vec3 pos,
                     glm::vec3 rel_pos, glm::vec3 scale, float mass,
                     btRigidBody *turret)
    : ConvexItem(std::move(name),
                 std::make_shared<ObjShape>(mgr, "obj/anubis_canon.obj"), pos,
                 scale, mass),
      angle(0.f) {

  btVector3 turret_pivot = btVector3(rel_pos.x, rel_pos.y, rel_pos.z);
  btVector3 canon_pivot = btVector3(0.f, 0.f, 0);
  btVector3 axis = btVector3(1, 0, 0);
  hinge = new btHingeConstraint(*turret, *ConvexItem::get_body(), turret_pivot,
                                canon_pivot, axis, axis, true);

  hinge->setLimit(angle, angle);
}

void CanonItem::on_input(const user_input &input) {
  angle += -input.right_joystick.y * 2e-2f;

  angle = angle > 1.f ? 1.f : angle;
  angle = angle < -1.f ? -1.f : angle;

  hinge->setLimit(angle * float(M_PI) * 0.2f, angle * float(M_PI) * 0.2f);
}

glm::vec3 CanonItem::pos() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 4, -20, 1);
}

glm::vec3 CanonItem::look() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 0, 1, 1);
}

glm::vec3 CanonItem::up() {
  btTransform tr = ConvexItem::get_body()->getWorldTransform();
  float tmp[16];
  tr.getOpenGLMatrix(tmp);
  glm::mat4 model_mat = glm::make_mat4(tmp);

  return model_mat * glm::vec4(0, 1, 0, 0);
}

std::vector<btTypedConstraint *> CanonItem::get_constraints() {
  auto constraints = Item::get_constraints();
  constraints.push_back(hinge);
  return constraints;
}
