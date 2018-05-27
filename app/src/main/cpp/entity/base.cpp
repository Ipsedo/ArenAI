//
// Created by samuel on 27/05/18.
//

#include "base.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Base::~Base() {
    delete defaultMotionState;
    delete collisionShape;
    delete rigidBody;
}

std::tuple<glm::mat4, glm::mat4> Base::getMatrixes(glm::mat4 pMatrix, glm::mat4 vMatrix) {
    btScalar tmp[16];
    defaultMotionState->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale);

    glm::mat4 mvMatrix = vMatrix * modelMatrix;
    glm::mat4 mvpMatrix = pMatrix * mvMatrix;

    return tuple<glm::mat4, glm::mat4>(mvpMatrix, mvMatrix);
}
