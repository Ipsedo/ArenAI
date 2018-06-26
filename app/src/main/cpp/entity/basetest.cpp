//
// Created by samuel on 26/06/18.
//

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "basetest.h"

BaseTest::BaseTest(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo, const ModelVBO &model,
				   const glm::vec3 &s) : scale(s), modelVBO(model),  btRigidBody(constructionInfo) {

}


void BaseTest::update() {

}

void BaseTest::decreaseLife(int toSub) {

}

bool BaseTest::isDead() {
	return false;
}

void BaseTest::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	btScalar tmp[16];
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	//getMotionState()->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	modelVBO.draw(mvpMatrix, mvMatrix, lighPos);
}

BaseTest::~BaseTest() {

}
