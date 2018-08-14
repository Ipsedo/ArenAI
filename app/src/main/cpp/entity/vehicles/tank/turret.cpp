//
// Created by samuel on 13/08/18.
//

#define GLM_ENABLE_EXPERIMENTAL

#include "turret.h"
#include "../../poly/cone.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../../utils/rigidbody.h"
#include "../../../utils/assets.h"

Turret::Turret(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
			   btDefaultMotionState *motionState, DiffuseModel *modelVBO,
			   const glm::vec3 &scale, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos)
		: Base(constructionInfo, motionState, modelVBO, scale),
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
		motionState->setWorldTransform(tr);
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

Canon::Canon(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
			 btDefaultMotionState *motionState, DiffuseModel *modelVBO,
			 const glm::vec3 &scale, btDynamicsWorld *world, Base *turret, btVector3 turretPos, DiffuseModel *missile)
		: Base(constructionInfo, motionState, modelVBO, scale),
		  angle(0.f), respawn(false), pos(turretPos + canonRelPos), hasClickedShoot(false), missile(missile) {

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
		motionState->setWorldTransform(tr);
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
	motionState->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	// Rotation matrix
	btQuaternion quat = motionState->m_graphicsWorldTrans.getRotation();
	btTransform tr;
	tr.setIdentity();
	tr.setRotation(quat);
	tr.getOpenGLMatrix(tmp);
	glm::mat4 rotMatrix = glm::make_mat4(tmp);

	glm::vec4 vec = modelMatrix * glm::vec4(0.f, 0.f, canonScale.z + 1.f, 1.f);

	rotMatrix = rotMatrix * glm::rotate(glm::mat4(1.f), 90.f, glm::vec3(1,0,0));

	Cone *cone = Cone::MakeCone(missile, vec, missileScale, rotMatrix, 10.f);

	glm::vec4 forceVec = modelMatrix * glm::vec4(0, 0, 500.f, 0);

	cone->applyCentralImpulse(btVector3(forceVec.x, forceVec.y, forceVec.z));

	entities->push_back(cone);
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

Turret *makeTurret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos) {
	std::string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");

	ModelVBO *modelVBO = new ModelVBO(turretObjTxt, turretColor);

	btCollisionShape *turretShape = parseObj(turretObjTxt);
	turretShape->setLocalScaling(btVector3(turretScale.x, turretScale.y, turretScale.z));

	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(chassisPos + turretRelPos);

	tuple<btRigidBody::btRigidBodyConstructionInfo, btDefaultMotionState *> cinfo = localCreateInfo(turretMass, tr, turretShape);
	return new Turret(get<0>(cinfo), get<1>(cinfo), modelVBO, turretScale, world, chassis, chassisPos);
}

Canon *makeCanon(AAssetManager *mgr, btDynamicsWorld *world, Base *turret, btVector3 turretPos) {
	std::string canonObjTxt = getFileText(mgr, "obj/cylinderZ.obj");

	ModelVBO *modelVBO = new ModelVBO(canonObjTxt, turretColor);

	btCollisionShape *canonShape = new btCylinderShapeZ(btVector3(canonScale.x, canonScale.y, canonScale.z));

	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(canonRelPos + turretPos);

	tuple<btRigidBody::btRigidBodyConstructionInfo, btDefaultMotionState *> cinfo
			= localCreateInfo(canonMass, tr, canonShape);

	return new Canon(get<0>(cinfo), get<1>(cinfo), modelVBO, canonScale, world, turret, turretPos,
		new ModelVBO(getFileText(mgr, "obj/cone.obj"),
					 new float[4]{(float) rand() / RAND_MAX,
								  (float) rand() / RAND_MAX,
								  (float) rand() / RAND_MAX,
								  1.f}));
}
