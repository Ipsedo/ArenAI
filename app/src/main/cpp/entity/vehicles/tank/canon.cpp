//
// Created by samuel on 10/09/18.
//

#include "canon.h"
#include "../../ammu/missile.h"
#include "../../../utils/vec.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

/////////////////////
// Canon
/////////////////////

#define LIMIT_ANGLE_CAMERA -float(M_PI) / 8.f

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
			   canonScale, glm::mat4(1.0f), canonMass, true),
		  angle(0.f), added(0), respawn(false), pos(turretPos + canonRelPos),
		  hasClickedShoot(false), missile(makeMissileModel(mgr)), maxFramesFire(25), fireCounter(0), turret(turret) {
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
	added = in.turretUp;
	respawn = in.respawn;
	hasClickedShoot = in.fire;
}

void Canon::update() {
	Base::update();

	angle += added * 5e-2f;
	angle = angle > 1.f ? 1.f : angle;
	angle = angle < -1.f ? -1.f : angle;
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

glm::mat4 Canon::getCamRot() {
	btScalar tmp1[16];
	btTransform tr1;
	getMotionState()->getWorldTransform(tr1);
	tr1.getOpenGLMatrix(tmp1);
	glm::mat4 canonModelMatrix = glm::make_mat4(tmp1);

	btScalar tmp2[16];
	btTransform tr2;
	turret->getMotionState()->getWorldTransform(tr2);
	tr2.getOpenGLMatrix(tmp2);
	glm::mat4 turretModelMatrix = glm::make_mat4(tmp2);

	glm::mat3 rot = hinge->getHingeAngle() < LIMIT_ANGLE_CAMERA ?
					glm::mat3(turretModelMatrix *
							  glm::rotate(glm::mat4(1.0f), LIMIT_ANGLE_CAMERA, glm::vec3(1.f, 0.f, 0.f))) :
					glm::mat3(canonModelMatrix);

	canonModelMatrix[0] = glm::vec4(rot[0], canonModelMatrix[0][3]);
	canonModelMatrix[1] = glm::vec4(rot[1], canonModelMatrix[1][3]);
	canonModelMatrix[2] = glm::vec4(rot[2], canonModelMatrix[2][3]);

	return canonModelMatrix;
}

glm::vec3 Canon::camPos(bool VR) {
	glm::mat4 modelMatrix = getCamRot();

	glm::vec4 p(0.f, 3.f, -12.f, 1.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Canon::camLookAtVec(bool VR) {
	glm::mat4 modelMatrix = getCamRot();

	glm::vec4 p(0.f, -0.2f, 1.f, 0.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

glm::vec3 Canon::camUpVec(bool VR) {
	glm::mat4 modelMatrix = getCamRot();

	glm::vec4 p(0.f, 1.f, 0.f, 0.f);
	p = modelMatrix * p;

	return glm::vec3(p.x, p.y, p.z);
}

Canon::~Canon() {
	delete missile;
}