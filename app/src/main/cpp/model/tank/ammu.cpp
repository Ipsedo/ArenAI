//
// Created by samuel on 13/04/2023.
//

#include "./ammu.h"

#include <utility>

std::shared_ptr<Shape> ShellItem::load_shape(AAssetManager *mgr) {
  return std::make_shared<ObjShape>(mgr, "obj/cone.obj");
}

ShellItem::ShellItem(AAssetManager *mgr, glm::vec3 pos, glm::quat rot,
                     glm::vec3 scale, float mass)
    : ConvexItem(ShellItem::NAME, ShellItem::load_shape(mgr), pos, scale,
                 mass) {

  btTransform shell_tr;
  shell_tr.setIdentity();
  shell_tr.setOrigin(btVector3(pos.x, pos.y, pos.z));
  shell_tr.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

  ConvexItem::get_body()->setWorldTransform(shell_tr);
}
