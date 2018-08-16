//
// Created by samuel on 26/06/18.
//

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "base.h"

Base::Base(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
				   DiffuseModel *model, const glm::vec3 &s, bool hasOwnModel)
		: scale(s), hasOwnModel(hasOwnModel), modelVBO(model), btRigidBody(constructionInfo) {

}


void Base::update() {

}

void Base::decreaseLife(int toSub) {

}

bool Base::isDead() {
	return false;
}

void Base::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	btScalar tmp[16];
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	modelVBO->draw(mvpMatrix, mvMatrix, lighPos);
}

Base::~Base() {
	btRigidBody::~btRigidBody();
	if (hasOwnModel)
		delete modelVBO;
}
