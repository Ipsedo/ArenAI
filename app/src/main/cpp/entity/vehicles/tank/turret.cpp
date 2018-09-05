//
// Created by samuel on 13/08/18.
//

#include "turret.h"

#include "../../ammu/missile.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../../utils/rigidbody.h"
#include "../../../utils/assets.h"
#include "../../../utils/vec.h"

/////////////////////
// Turret
/////////////////////

ModelVBO *makeTurretModel(AAssetManager *mgr) {
	string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");
	return new ModelVBO(turretObjTxt, turretColor[0], turretColor[1], turretColor[2], turretColor[3]);
}

Turret::Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos)
		: Poly([mgr](glm::vec3 scale) {
								   string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");
								   btCollisionShape *turretShape = parseObj(turretObjTxt);
								   turretShape->setLocalScaling(btVector3(turretScale.x, turretScale.y, turretScale.z));
								   return turretShape;
							   },
			   makeTurretModel(mgr),
			   btVector3ToVec3(chassisPos + turretRelPos), turretScale, glm::mat4(1.0f),  turretMass, true),
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
	return new ModelVBO(canonObjTxt, turretColor[0], turretColor[1], turretColor[2], turretColor[3]);
}

ModelVBO *makeMissileModel(AAssetManager *mgr) {
	return new ModelVBO(getFileText(mgr, "obj/cone.obj"), 0.2f, 0.7f, 0.05f, 1.f);
}

Canon::Canon(AAssetManager *mgr, btDynamicsWorld *world, Base *turret, btVector3 turretPos)
		: Poly([mgr](glm::vec3 scale) {
				   string canonObjTxt = getFileText(mgr, "obj/cylinderZ.obj");
				   return new btCylinderShapeZ(btVector3(canonScale.x, canonScale.y, canonScale.z));
			   },
			   makeCanonModel(mgr), btVector3ToVec3(turretPos + canonRelPos),
			   canonScale, glm::mat4(1.0f),  canonMass, true),
		  angle(0.f), respawn(false), pos(turretPos + canonRelPos), hasClickedShoot(false),
		  missile(makeMissileModel(mgr)), maxFramesFire(25), fireCounter(0) {
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
	angle = in.turretUp;
	respawn = in.respawn;
	hasClickedShoot = in.fire;
}

void Canon::update() {
	Base::update();

	hinge->setLimit(angle * float(M_PI) * 0.2f, angle * float(M_PI) * 0.2f);

	fireCounter = fireCounter + 1 <= maxFramesFire ? fireCounter + 1 : maxFramesFire;

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

vector<Base *> Canon::fire() {
	if (!hasClickedShoot || fireCounter < maxFramesFire) {
		return vector<Base *>();
	}
	fireCounter = 0;
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
	glm::quat q(quat.w(), quat.x(), quat.y(), quat.z());

	glm::mat4 rotMat = glm::mat4_cast(q);
	rotMat = rotMat * glm::rotate(glm::mat4(1.f), float(M_PI / 2.f), glm::vec3(1, 0, 0));

	glm::vec4 vec = modelMatrix * glm::vec4(0.f, 0.f, canonScale.z + 1.f, 1.f);

	Missile *m = new Missile(missile, glm::vec3(vec), missileScale, rotMat, 10.f, 1);

	glm::vec4 forceVec = modelMatrix * glm::vec4(0, 0, 500.f, 0);
	m->applyCentralImpulse(btVector3(forceVec.x, forceVec.y, forceVec.z));

	vector<Base *> res;
	res.push_back(m);
	return res;
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