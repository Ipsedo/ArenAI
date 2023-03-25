//
// Created by samuel on 19/03/2023.
//

#include "./item.h"

#include <glm/gtc/type_ptr.hpp>

#include "../utils/asset.h"
#include "../utils/logging.h"

Item::Item(std::string name) : name(std::move(name)) {}

std::string Item::get_name() { return name; }

glm::mat4 Item::get_model_matrix() {
  btScalar tmp[16];
  btTransform tr;

  get_body()->getMotionState()->getWorldTransform(tr);

  tr.getOpenGLMatrix(tmp);

  return glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), _get_scale());
}

nlohmann::json Item::get_state() {
  btRigidBody *body = get_body();
  btTransform transform = body->getWorldTransform();

  btVector3 pos = transform.getOrigin();
  btQuaternion rot_quat = transform.getRotation();

  nlohmann::json state = {
      {"pos", {{"x", pos.x()}, {"y", pos.y()}, {"z", pos.z()}}},
      {"rot_quat",
       {{"x", rot_quat.x()},
        {"y", rot_quat.y()},
        {"z", rot_quat.z()},
        {"w", rot_quat.w()}

       }},
      {"mass", float(body->getMass())}};

  return state;
}
