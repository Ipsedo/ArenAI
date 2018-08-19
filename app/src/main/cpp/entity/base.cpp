//
// Created by samuel on 26/06/18.
//

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "base.h"

int nbNew;
int nbDel;

Base::Base(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
		   DiffuseModel *model, const glm::vec3 &s, bool hasOwnModel)
		: scale(s), hasOwnModel(hasOwnModel), modelVBO(model), btRigidBody(constructionInfo) {nbNew++;}


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

	modelVBO->draw(mvpMatrix, mvMatrix, infos.light_pos);
}

Base::~Base() {
	nbDel++;
	btRigidBody::~btRigidBody();
	if (hasOwnModel)
		delete modelVBO;
}
