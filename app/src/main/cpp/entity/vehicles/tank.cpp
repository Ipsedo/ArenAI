//
// Created by samuel on 30/05/18.
//

#include <android/asset_manager.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include "tank.h"

static float wheelRadius = 0.7f;
static float wheelWidth = 0.4f;
static float chassisMass = 850.f;
static float wheelMass = 10.f;
static float turretMass = 150.f;

Tank::Tank(btDynamicsWorld *world, AAssetManager *mgr) {
	direction = 0.f;
	speed = 0.f;
	turretDir = 0.f;
	init(world, mgr);
}

void Tank::onInput(input in) {
	int motorAxis = 5;
	float attenuation = 10.f;

	direction += in.xAxis / attenuation;
	if (direction > 1.f) direction = 1.f;
	if (direction < -1.f) direction = -1.f;

	pHinge2[0]->setLimit(
			motorAxis,
			float(M_PI) * direction / 6.f,
			float(M_PI) * direction / 6.f);
	pHinge2[1]->setLimit(
			motorAxis,
			float(M_PI) * direction / 6.f,
			float(M_PI) * direction / 6.f);

	pHinge2[4]->setLimit(
			motorAxis,
			-float(M_PI) * direction / 6.f,
			-float(M_PI) * direction / 6.f);
	pHinge2[5]->setLimit(
			motorAxis,
			-float(M_PI) * direction / 6.f,
			-float(M_PI) * direction / 6.f);

	motorAxis = 3;
	speed += in.speed / attenuation;
	if (speed > 1.f) speed = 1.f;
	if (speed < -1.f) speed = -1.f;
	if (in.brake) speed = 0.f;

	for (int i = 0; i < 6; i++) {
		pHinge2[i]->setTargetVelocity(motorAxis, -speed * 10.f);
	}

	motorAxis = 5;
	attenuation = 100.f;
	turretDir += in.turretDir / attenuation;
	if (turretDir > 1.f) turretDir = 1.f;
	if (turretDir < -1.f) turretDir = -1.f;
	pHinge2[6]->setLimit(motorAxis, turretDir * float(M_PI) * 2.f, turretDir * float(M_PI) * 2.f);
}

void Tank::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	btScalar tmp[16];

	// Chassis
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[0]);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	modelVBOs[0]->draw(mvpMatrix, mvMatrix, lighPos);

	//Wheels
	int idxDebutWheels = 1;
	for (int i = 0; i < 6; i++) {
		int idx = idxDebutWheels + i;
		defaultMotionState[idx]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
		modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[1]);

		mvMatrix = vMatrix * modelMatrix;
		mvpMatrix = pMatrix * mvMatrix;

		modelVBOs[1]->draw(mvpMatrix, mvMatrix, lighPos);
	}

	defaultMotionState[7]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[2]);

	mvMatrix = vMatrix * modelMatrix;
	mvpMatrix = pMatrix * mvMatrix;

	modelVBOs[2]->draw(mvpMatrix, mvMatrix, lighPos);
}

// TODO cam selon tourelle
glm::vec3 Tank::camPos() {
	btScalar tmp[16];

	// Chassis
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, 3.f, 0.f, 1.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

glm::vec3 Tank::camLookAtVec() {
	btScalar tmp[16];

	// Chassis
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, 0.f, 1.f, 0.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

glm::vec3 Tank::camUpVec() {
	btScalar tmp[16];

	// Chassis
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, 1.f, 0.f, 0.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

Tank::~Tank() {

}

/**
 * collisionShape[0] -> chassis
 * collisionShape[1-6] -> wheels
 *
 * rigidBody[0] -> chassis
 * rigidBody[1-6] -> wheels
 *
 * pHinge2[0-5] -> wheels
 * pHinge2[6] -> turret
 *
 * modelVBOs[0] -> chassis
 * modelVBOs[1] -> wheels
 *
 */
void Tank::init(btDynamicsWorld *world, AAssetManager *mgr) {
	std::string cubeObjTxt = getFileText(mgr, "obj/cube.obj");

	float posY = 5.f;

	btTransform tr;
	tr.setIdentity();

	btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
	collisionShape.push_back(chassisShape);

	tr.setOrigin(btVector3(0, posY + 0.f, 0));

	std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(chassisMass, tr,
																				 chassisShape);//chassisShape);
	btRigidBody *m_carChassis = std::get<0>(tmp);
	btDefaultMotionState *m_carChassiMotionState = std::get<1>(tmp);

	modelVBOs.push_back(new ModelVBO(cubeObjTxt, new float[4]{1.f, 0.f, 0.f, 1.f}));
	scale.push_back(glm::vec3(1.f, 0.5f, 2.f));
	rigidBody.push_back(m_carChassis);
	defaultMotionState.push_back(m_carChassiMotionState);

	btCollisionShape *m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));
	collisionShape.push_back(m_wheelShape);

	btVector3 wheelPos[6] = {
			btVector3(btScalar(-1.), btScalar(posY - 0.4), btScalar(1.75)),
			btVector3(btScalar(1.), btScalar(posY - 0.4), btScalar(1.75)),
			btVector3(btScalar(-1.), btScalar(posY - 0.4), btScalar(0)),
			btVector3(btScalar(1.), btScalar(posY - 0.4), btScalar(0)),
			btVector3(btScalar(1.), btScalar(posY - 0.4), btScalar(-1.75)),
			btVector3(btScalar(-1.), btScalar(posY - 0.4), btScalar(-1.75))
	};

	std::string cylObjText = getFileText(mgr, "obj/cylinderX.obj");
	scale.push_back(glm::vec3(wheelWidth, wheelRadius, wheelRadius));
	modelVBOs.push_back(new ModelVBO(cylObjText, new float[4]{0.f, 1.f, 0.f, 1.f}));

	for (int i = 0; i < 6; i++) {

		btRigidBody *pBodyA = rigidBody[0];
		pBodyA->setActivationState(DISABLE_DEACTIVATION);

		tr.setIdentity();
		tr.setOrigin(wheelPos[i]);

		tmp = localCreateRigidBody(wheelMass, tr, m_wheelShape);

		btRigidBody *pBodyB = std::get<0>(tmp);
		btDefaultMotionState *pBodyBMotionState = std::get<1>(tmp);

		rigidBody.push_back(pBodyB);
		defaultMotionState.push_back(pBodyBMotionState);

		pBodyB->setFriction(1110);
		pBodyB->setActivationState(DISABLE_DEACTIVATION);

		btVector3 parentAxis(0.f, 1.f, 0.f);
		btVector3 childAxis(1.f, 0.f, 0.f);
		btVector3 anchor = tr.getOrigin();
		pHinge2.push_back(new btHinge2Constraint(*pBodyA, *pBodyB, anchor, parentAxis,
												 childAxis));

		pHinge2[i]->setLowerLimit(float(-M_PI) * 0.5f);
		pHinge2[i]->setUpperLimit(float(M_PI) * 0.5f);

		world->addConstraint(pHinge2[i], true);

		/**
		 * Axis :
		 * 	- Linear, y = 2
		 * 	- Rotation, x = 3, z = 4, y = 5
		 */

		{
			int motorAxis = 3;
			pHinge2[i]->enableMotor(motorAxis, true);
			pHinge2[i]->setMaxMotorForce(motorAxis, 1e10f);
			pHinge2[i]->setTargetVelocity(motorAxis, 0);
		}

		{
			int motorAxis = 5;
			pHinge2[i]->setLimit(motorAxis, 0, 0);
		}

		{
			int index = 2;
			pHinge2[i]->setLimit(index, -0.4f, 0.2f);
			pHinge2[i]->setDamping(index, 2.3f, true);
			pHinge2[i]->setStiffness(index, 20.f, true);
			pHinge2[i]->setBounce(index, 0.1f);
		}

		pHinge2[i]->setDbgDrawSize(btScalar(5.f));
	}

	{
		// Bug
		pHinge2[2]->setLimit(0, 0.f, 0.f);
		pHinge2[3]->setLimit(0, 0.f, 0.f);
		pHinge2[2]->setLimit(1, 0.f, 0.f);
		pHinge2[3]->setLimit(1, 0.f, 0.f);
		pHinge2[2]->setLimit(2, 0.f, 0.f);
		pHinge2[3]->setLimit(2, 0.f, 0.f);
	}

	// Tourelle
	scale.push_back(glm::vec3(0.75f, 0.25f, 0.75f));

	std::string coneObjTxt = getFileText(mgr, "obj/cone.obj");

	modelVBOs.push_back(new ModelVBO(coneObjTxt, new float[4]{0.f, 0.8f, 0.8f, 1.f}));

	btCollisionShape* turretShape = new btConeShape(1.f, 2.f);
	turretShape->setLocalScaling(btVector3(scale[2].x, scale[2].y, scale[2].z));

	tr.setIdentity();
	tr.setOrigin(btVector3(0.f, posY + 0.5f + scale[2].y, 0.f));

	tmp = localCreateRigidBody(turretMass, tr, turretShape);

	btRigidBody *pBodyA = rigidBody[0];
	btRigidBody *pBodyB = std::get<0>(tmp);
	btDefaultMotionState *pBodyBMotionState = std::get<1>(tmp);

	rigidBody.push_back(pBodyB);
	defaultMotionState.push_back(pBodyBMotionState);

	btVector3 parentAxis(0.f, 1.f, 0.f);
	btVector3 childAxis(1.f, 0.f, 0.f);
	btVector3 anchor = tr.getOrigin();
	pHinge2.push_back(new btHinge2Constraint(*pBodyA, *pBodyB, anchor, parentAxis,
											 childAxis));
	unsigned long indexHingeTurret = pHinge2.size() - 1;

	world->addConstraint(pHinge2[indexHingeTurret], true);

	for (int i = 0; i < 6; i++) {
		pHinge2[indexHingeTurret]->setLimit(i, 0, 0);
	}
}
