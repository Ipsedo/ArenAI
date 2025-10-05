//
// Created by samuel on 26/06/18.
//

#include "base.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

Base::Base(
    const btRigidBody::btRigidBodyConstructionInfo &constructionInfo, GLDrawable *drawable,
    const glm::vec3 &s, bool hasOwnModel)
    : scale(s), hasOwnModel(hasOwnModel), drawable(drawable), btRigidBody(constructionInfo) {}

void Base::update() {}

void Base::decreaseLife(int toSub) {}

bool Base::isDead() { return false; }

void Base::draw(draw_infos infos) {
    btScalar tmp[16];
    btTransform tr;
    getMotionState()->getWorldTransform(tr);
    tr.getOpenGLMatrix(tmp);
    glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale);

    glm::mat4 mvMatrix = infos.view_matrix * modelMatrix;
    glm::mat4 mvpMatrix = infos.proj_matrix * mvMatrix;

    gl_draw_info gl_info{mvpMatrix, mvMatrix, infos.light_pos, infos.camera_pos};

    drawable->draw(gl_info);
}

bool Base::needExplosion() { return true; }

void Base::onContactFinish(Base *other) {}

Base::~Base() {
    btRigidBody::~btRigidBody();
    if (hasOwnModel) delete drawable;
}
