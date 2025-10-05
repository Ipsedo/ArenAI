//
// Created by samuel on 13/08/18.
//

#include "chassis.h"

#include "../../graphics/drawable/normalmodel.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include "../../utils/vec.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

NormalMapModel *makeChassisModel(AAssetManager *mgr) {
  return new NormalMapModel(
    mgr, "obj/tank_chassis.obj", "textures/chassis_tex.png", "textures/158_norm.png");
}

ModelVBO *f(AAssetManager *mgr) {
  return new ModelVBO(getFileText(mgr, "obj/tank_chassis.obj"), 1.f, 0.f, 0.f, 1.f);
}

Chassis::Chassis(AAssetManager *mgr, btVector3 pos)
    : Poly(
      [mgr](glm::vec3 scale) {
        string objTxt = getFileText(mgr, "obj/tank_chassis.obj");
        btCollisionShape *shape = parseObj(objTxt);
        shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
        return shape;
      },
      makeChassisModel(mgr), glm::vec3(pos.x(), pos.y(), pos.z()), chassisScale, glm::mat4(1.0f),
      chassisMass, true),
      respawn(false), pos(pos), isHit(false) {
  /*btVector3 centerOfMass = getCenterOfMassPosition();
  btTransform tr;
  tr.setIdentity();
  tr.setOrigin(centerOfMass + btVector3(0, -chassisScale.y, 0));
  setCenterOfMassTransform(tr);*/
}

void Chassis::onInput(input in) { respawn = in.respawn; }

void Chassis::update() {
  Base::update();

  if (respawn) {
    btTransform tr;
    tr.setIdentity();
    tr.setOrigin(pos);

    clearForces();
    setLinearVelocity(btVector3(0, 0, 0));
    setAngularVelocity(btVector3(0, 0, 0));

    getMotionState()->setWorldTransform(tr);
    setWorldTransform(tr);

    respawn = false;
  }
}

glm::vec3 Chassis::camLookAtVec(bool VR) {
  btScalar tmp[16];

  btTransform tr;
  getMotionState()->getWorldTransform(tr);
  tr.getOpenGLMatrix(tmp);
  glm::mat4 modelMatrix = glm::make_mat4(tmp);

  glm::vec4 p(0.f, 0.f, 1.f, 0.f);
  p = modelMatrix * p;

  return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Chassis::camUpVec(bool VR) {
  btScalar tmp[16];

  btTransform tr;
  getMotionState()->getWorldTransform(tr);
  tr.getOpenGLMatrix(tmp);
  glm::mat4 modelMatrix = glm::make_mat4(tmp);

  glm::vec4 p(0.f, 1.f, 0.f, 0.f);
  p = modelMatrix * p;

  return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Chassis::camPos(bool VR) {
  btScalar tmp[16];

  btTransform tr;
  getMotionState()->getWorldTransform(tr);
  tr.getOpenGLMatrix(tmp);
  glm::mat4 modelMatrix = glm::make_mat4(tmp);

  glm::vec4 p(0.f, 2.f, 0.f, 1.f);
  p = modelMatrix * p;

  return glm::vec3(p.x, p.y, p.z);
}

output Chassis::getOutput() {
  bool vibrate = isHit;
  isHit = false;
  return {vibrate};
}

void Chassis::decreaseLife(int toSub) {
  isHit = toSub > 0;
  Base::decreaseLife(toSub);
}
