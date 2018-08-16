//
// Created by samuel on 13/08/18.
//

#define GLM_ENABLE_EXPERIMENTAL

#include "turret.h"

#include "../../missile.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../../utils/rigidbody.h"
#include "../../../utils/assets.h"
#include "../../../utils/vec.h"

ModelVBO *makeTurretModel(AAssetManager *mgr) {
	string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");
	return new ModelVBO(turretObjTxt, turretColor);
}

Turret::Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos)
		: Poly(Poly::makeCInfo([mgr](glm::vec3 scale) {
								   string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");
								   btCollisionShape *turretShape = parseObj(turretObjTxt);
								   turretShape->setLocalScaling(btVector3(turretScale.x, turretScale.y, turretScale.z));
								   return turretShape;
							   },
							   btVector3ToVec3(chassisPos + turretRelPos), glm::mat4(1.0f), turretScale, turretMass),
			   makeTurretModel(mgr), turretScale, true),
		  angle(0.f), respawn(false), pos(chassisPos + turretRelPos) {
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(pos);

	btRigidBody *pBodyA = chassis;
	btRigidBody *pBodyB = this;

	btVector3 pivotA = btVector3(0.f, chassisScale.y, 0.f);
	btVector3 pivotB = btVector3(0.f, -turretScale.y, 0.f);
	btVector3 axis = btVector3(0.f, 1.f, 0.f);

	hinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
	world->addConstraint(hinge, true);
	hinge->setLimit(0, 0);
}

void Turret::onInput(input in) {
	angle = in.turretDir;
	respawn = in.respawn;
}

void Turret::update() {
	Base::update();
	hinge->setLimit(angle * (float) M_PI * 0.5f, angle * (float) M_PI * 0.5f);

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


/////////////////////
// Canon
/////////////////////

ModelVBO *makeCanonModel(AAssetManager *mgr) {
	string canonObjTxt = getFileText(mgr, "obj/cylinderZ.obj");
	return new ModelVBO(canonObjTxt, turretColor);
}

ModelVBO *makeMissileModel(AAssetManager *mgr) {
	return new ModelVBO(getFileText(mgr, "obj/cone.obj"), new float[4]{0.2f, 0.7f, 0.05f, 1.f});
}

Canon::Canon(AAssetManager *mgr, btDynamicsWorld *world, Base *turret, btVector3 turretPos)
		: Poly(Poly::makeCInfo([mgr](glm::vec3 scale) {
								   string canonObjTxt = getFileText(mgr, "obj/cylinderZ.obj");
								   return new btCylinderShapeZ(btVector3(canonScale.x, canonScale.y, canonScale.z));
							   }, btVector3ToVec3(turretPos + canonRelPos), glm::mat4(1.0f), canonScale, canonMass),
			   makeCanonModel(mgr), canonScale, true),
		  angle(0.f), respawn(false), pos(turretPos + canonRelPos), hasClickedShoot(false), missile(makeMissileModel(mgr)) {
	btRigidBody *pBodyA = turret;
	btRigidBody *pBodyB = this;

	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(canonRelPos + turretPos + btVector3(0, 0, -canonScale.z));

	btVector3 pivotA = btVector3(0.f, 0.f, turretScale.z - canonOffset);
	btVector3 pivotB = btVector3(0.f, 0.f, -canonScale.z);
	btVector3 axis = btVector3(1, 0, 0);
	hinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
	world->addConstraint(hinge, true);
	hinge->setLimit(angle, angle);
}

void Canon::onInput(input in) {
	angle = -in.turretUp;
	respawn = in.respawn;
	hasClickedShoot = in.fire;
}

void Canon::update() {
	Base::update();
	hinge->setLimit(angle * float(M_PI) * 0.2f, angle * float(M_PI) * 0.2f);

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

void Canon::fire(std::vector<Base *> *entities) {
	if (!hasClickedShoot) {
		return;
	}
	hasClickedShoot = false;

	btScalar tmp[16];
	glm::vec3 missileScale = glm::vec3(0.1f, 0.3f, 0.1f);

	// Model matrix
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	// Rotation matrix
	tr.setIdentity();
	getMotionState()->getWorldTransform(tr);
	btQuaternion quat = tr.getRotation();
	tr.setIdentity();
	tr.setRotation(quat);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 rotMatrix = glm::make_mat4(tmp);

	glm::vec4 vec = modelMatrix * glm::vec4(0.f, 0.f, canonScale.z + 1.f, 1.f);

	rotMatrix = rotMatrix * glm::rotate(glm::mat4(1.f), 90.f, glm::vec3(1, 0, 0));

	Missile *m = new Missile(missile, vec, missileScale, rotMatrix, 10.f, 10);

	glm::vec4 forceVec = modelMatrix * glm::vec4(0, 0, 500.f, 0);

	m->applyCentralImpulse(btVector3(forceVec.x, forceVec.y, forceVec.z));

	entities->push_back(m);
}

glm::vec3 Canon::camPos(bool VR) {
	btScalar tmp[16];
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 p(0.f, 3.f, -12.f, 1.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Canon::camLookAtVec(bool VR) {
	btScalar tmp[16];
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 p(0.f, -0.2f, 1.f, 0.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Canon::camUpVec(bool VR) {
	btScalar tmp[16];
	btTransform tr;
	getMotionState()->getWorldTransform(tr);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 p(0.f, 1.f, 0.f, 0.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

Canon::~Canon() {
	delete missile;
}