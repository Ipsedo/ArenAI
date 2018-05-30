//
// Created by samuel on 27/05/18.
//

#include "base.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Base::~Base() {
	for (btDefaultMotionState *state : defaultMotionState)
		delete state;
	for (btCollisionShape *shape : collisionShape)
		delete shape;
	for (btRigidBody *body : rigidBody)
		delete body;
}

std::tuple<glm::mat4, glm::mat4> Base::getMatrixes(glm::mat4 pMatrix, glm::mat4 vMatrix) {
	btScalar tmp[16];
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[0]);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	return tuple<glm::mat4, glm::mat4>(mvpMatrix, mvMatrix);
}
