//
// Created by samuel on 13/08/18.
//

#include "chassis.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../../utils/assets.h"
#include "../../../utils/rigidbody.h"

Chassis::Chassis(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
				 ModelVBO *modelVBO, const glm::vec3 &scale, const btVector3 centerPos)
		: Base(constructionInfo, modelVBO, scale),
		  respawn(false), pos(centerPos) {
}

void Chassis::onInput(input in) {
	respawn = in.respawn;
}

void Chassis::update() {
	Base::update();

	if (respawn) {
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(pos);

		getMotionState()->setWorldTransform(tr);
		setWorldTransform(tr);

		clearForces();
		setLinearVelocity(btVector3(0, 0, 0));
		setAngularVelocity(btVector3(0, 0, 0));

		respawn = false;
	}
}

Chassis *makeChassis(AAssetManager *mgr, btVector3 pos) {
	std::string chassisObjTxt = getFileText(mgr, "obj/tank_chassis.obj");

	ModelVBO *modelVBO = new ModelVBO(chassisObjTxt, chassisColor);
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(pos);

	btCollisionShape *chassisShape = parseObj(chassisObjTxt);
	btRigidBody::btRigidBodyConstructionInfo cinfo = localCreateInfo(chassisMass, tr, chassisShape);

	return new Chassis(cinfo, modelVBO, chassisScale, pos);
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


