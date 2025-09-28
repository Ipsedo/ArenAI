//
// Created by samuel on 19/03/2023.
//

#include <phyvr_model/item.h>

#include <glm/gtc/type_ptr.hpp>

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

  btVector3 inertia = body->getLocalInertia();
  btVector3 vel = body->getLinearVelocity();
  btVector3 ang_vel = body->getAngularVelocity();

  return {
      {"name", name},
      {"pos", {{"x", pos.x()}, {"y", pos.y()}, {"z", pos.z()}}},
      {"rot_quat",
       {"x", rot_quat.x()},
       {"y", rot_quat.y()},
       {"z", rot_quat.z()},
       {"w", rot_quat.w()}},
      {"inertia", {"x", inertia.x()}, {"y", inertia.y()}, {"z", inertia.z()}},
      {"vel", {"x", vel.x()}, {"y", vel.y()}, {"z", vel.z()}},
      {"ang_vel", {"x", ang_vel.x()}, {"y", ang_vel.y()}, {"z", ang_vel.z()}},
      {"friction", float(body->getFriction())},
      {"mass", float(body->getMass())}};
}

void Item::from_state(const nlohmann::json &state) {
  name = state["name"];

  btVector3 pos(state["pos"]["x"], state["pos"]["y"], state["pos"]["z"]);
  btQuaternion rot_quat(state["rot_quat"]["x"], state["rot_quat"]["y"],
                        state["rot_quat"]["z"], state["rot_quat"]["w"]);
  btVector3 inertia(state["inertia"]["x"], state["inertia"]["y"],
                    state["inertia"]["z"]);
  btVector3 vel(state["vel"]["x"], state["vel"]["y"], state["vel"]["z"]);
  btVector3 ang_vel(state["ang_vel"]["x"], state["ang_vel"]["y"],
                    state["ang_vel"]["z"]);
  float friction = state["friction"];
  float mass = state["mass"];

  btRigidBody *body = get_body();

  btTransform transform;
  transform.setIdentity();
  transform.setOrigin(pos);
  transform.setRotation(rot_quat);

  body->setWorldTransform(transform);

  body->setAngularVelocity(ang_vel);
  body->setLinearVelocity(vel);

  body->setMassProps(mass, inertia);
  body->setFriction(friction);
}

std::vector<btTypedConstraint *> Item::get_constraints() { return {}; }
