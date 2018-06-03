//
// Created by samuel on 30/05/18.
//

#define GLM_ENABLE_EXPERIMENTAL
#include <android/asset_manager.h>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include "../missile.h"
#include "../poly/box.h"
#include "tank.h"

static float wheelRadius = 0.8f;
static float wheelWidth = 0.4f;
static float wheelOffset = 0.6f;
static float wheelbaseOffset = 0.1f;

static float chassisMass = 40000.f;
static float wheelMass = 300.f;
static float turretMass = 100.f;
static float canonMass = 10.f;

static float canonOffset = 0.1f;

static float wheelColor[4]{52.f / 255.f, 73.f / 255.f, 94.f / 255.f, 1.f};
static float chassisColor[4]{150.f / 255.f, 40.f / 255.f, 27.f / 255.f, 1.f};
static float turretColor[4]{4.f / 255.f, 147.f / 255.f, 114.f / 255.f, 1.f};

Tank::Tank(glm::vec3 pos, btDynamicsWorld *world, AAssetManager *mgr, vector<Base*>* bases) {
	std::string objTxt = getFileText(mgr, "obj/cone.obj");

	hasClickedShoot = false;

	this->world = world;

	missile = new ModelVBO(objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});

	this->bases = bases;

	spawnPos = pos;

	direction = 0.f;
	speed = 0.f;

	chassisScale = glm::vec3(1.2f, 0.5f, 2.f);
	turretScale = glm::vec3(0.9f, 0.25f, 1.2f);
	wheelScale = glm::vec3(wheelWidth, wheelRadius, wheelRadius);
	canonScale = glm::vec3(0.1f, 0.1f, 0.8f);

	turretUp = 0.f;
	turretDir = 0.f;
	turretPos = spawnPos + glm::vec3(0.f, chassisScale.y + turretScale.y, 0.f);
	canonPos = turretPos + glm::vec3(0.f, 0.f, turretScale.z + canonScale.z - canonOffset);

	nbWheel = 6;

	wheelPos = new btVector3[nbWheel]{
			btVector3(spawnPos.x - (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z + 1.75f),
			btVector3(spawnPos.x + (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z + 1.75f),
			btVector3(spawnPos.x - (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z + 0),
			btVector3(spawnPos.x + (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z + 0),
			btVector3(spawnPos.x + (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z - 1.75f),
			btVector3(spawnPos.x - (chassisScale.x + wheelbaseOffset), spawnPos.y - wheelOffset, spawnPos.z - 1.75f)
	};

	init(world, mgr);
}

void Tank::onInput(input in) {
	directionAdd = in.xAxis;
	speedAdd = in.speed;
	turretDirAdd = in.turretDir;
	turretUpAdd = in.turretUp;

	if (in.brake) {
		speedAdd = 0;
		speed = 0.f;
	}

	if (in.respawn) respawn();
	if (in.fire) hasClickedShoot = true;
}

void Tank::update() {
	float attenuation = 10.f;

	direction += directionAdd / attenuation;
	if (direction > 1.f) direction = 1.f;
	if (direction < -1.f) direction = -1.f;

	speed += speedAdd / attenuation;
	if (speed > 1.f) speed = 1.f;
	if (speed < -1.f) speed = -1.f;

	// Turret rotation
	attenuation = 50.f;
	turretDir += turretDirAdd / attenuation;
	if (turretDir > 1.f) turretDir = 1.f;
	if (turretDir < -1.f) turretDir = -1.f;

	// Canon elevation
	attenuation = 30.f;
	turretUp += turretUpAdd / attenuation;
	if (turretUp > 1.f) turretUp = 1.f;
	if (turretUp < -1.f) turretUp = -1.f;

	// Wheel direction
	int motorAxis = 5;
	wheelHinge2[0]->setLimit(
			motorAxis,
			float(M_PI) * direction / 6.f,
			float(M_PI) * direction / 6.f);
	wheelHinge2[1]->setLimit(
			motorAxis,
			float(M_PI) * direction / 6.f,
			float(M_PI) * direction / 6.f);

	wheelHinge2[4]->setLimit(
			motorAxis,
			-float(M_PI) * direction / 6.f,
			-float(M_PI) * direction / 6.f);
	wheelHinge2[5]->setLimit(
			motorAxis,
			-float(M_PI) * direction / 6.f,
			-float(M_PI) * direction / 6.f);

	motorAxis = 3;
	for (int i = 0; i < nbWheel; i++)
		wheelHinge2[i]->setTargetVelocity(motorAxis, -speed * 10.f);

	turretHinge->setLimit(turretDir * float(M_PI) * 0.5f, turretDir * float(M_PI) * 0.5f);

	canonHinge->setLimit(turretUp * float(M_PI) * 0.2f, turretUp * float(M_PI) * 0.2f);
}

void Tank::respawn() {
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(spawnPos.x, spawnPos.y, spawnPos.z));
	defaultMotionState[0]->setWorldTransform(tr);
	rigidBody[0]->setWorldTransform(tr);
	rigidBody[0]->clearForces();
	rigidBody[0]->setLinearVelocity(btVector3(0,0,0));
	rigidBody[0]->setAngularVelocity(btVector3(0,0,0));

	for (int i = 0; i < nbWheel; i++) {
		int id = i + 1;

		tr.setIdentity();
		tr.setOrigin(wheelPos[i]);

		defaultMotionState[id]->setWorldTransform(tr);
		rigidBody[id]->setWorldTransform(tr);
		rigidBody[id]->clearForces();
		rigidBody[id]->setLinearVelocity(btVector3(0,0,0));
		rigidBody[id]->setAngularVelocity(btVector3(0,0,0));
	}

	tr.setIdentity();
	tr.setOrigin(btVector3(turretPos.x, turretPos.y, turretPos.z));
	defaultMotionState[7]->setWorldTransform(tr);
	rigidBody[7]->setWorldTransform(tr);
	rigidBody[7]->clearForces();
	rigidBody[7]->setLinearVelocity(btVector3(0,0,0));
	rigidBody[7]->setAngularVelocity(btVector3(0,0,0));

	tr.setIdentity();
	tr.setOrigin(btVector3(canonPos.x, canonPos.y, canonPos.z));
	defaultMotionState[8]->setWorldTransform(tr);
	rigidBody[8]->setWorldTransform(tr);
	rigidBody[8]->clearForces();
	rigidBody[8]->setLinearVelocity(btVector3(0,0,0));
	rigidBody[8]->setAngularVelocity(btVector3(0,0,0));
}

void Tank::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	btScalar tmp[16];

	// Chassis
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), chassisScale);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	modelVBOs[0]->draw(mvpMatrix, mvMatrix, lighPos);

	//Wheels
	int idxDebutWheels = 1;
	for (int i = 0; i < nbWheel; i++) {
		int idx = idxDebutWheels + i;
		defaultMotionState[idx]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
		modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), wheelScale);

		mvMatrix = vMatrix * modelMatrix;
		mvpMatrix = pMatrix * mvMatrix;

		modelVBOs[1]->draw(mvpMatrix, mvMatrix, lighPos);
	}

	// Turret
	defaultMotionState[7]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), turretScale);

	mvMatrix = vMatrix * modelMatrix;
	mvpMatrix = pMatrix * mvMatrix;

	modelVBOs[2]->draw(mvpMatrix, mvMatrix, lighPos);

	// Canon
	defaultMotionState[8]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), canonScale);

	mvMatrix = vMatrix * modelMatrix;
	mvpMatrix = pMatrix * mvMatrix;

	modelVBOs[3]->draw(mvpMatrix, mvMatrix, lighPos);
}

// TODO cam selon tourelle ?
glm::vec3 Tank::camPos(bool VR) {
	btScalar tmp[16];

	int id = VR ? 0 : 8;
	// Chassis : Canon
	defaultMotionState[id]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, 3.f, VR ? -1.f : -12.f, 1.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

glm::vec3 Tank::camLookAtVec(bool VR) {
	btScalar tmp[16];

	int id = VR ? 0 : 8;
	// Chassis : Canon
	defaultMotionState[id]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, VR ? 0.f : -0.2f, 1.f, 0.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

glm::vec3 Tank::camUpVec(bool VR) {
	btScalar tmp[16];

	int id = VR ? 0 : 8;
	// Chassis : Canon
	defaultMotionState[id]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 pos(0.f, 1.f, 0.f, 0.f);
	pos = modelMatrix * pos;

	return glm::vec3(pos.x, pos.y, pos.z);
}

Tank::~Tank() {
	for (ModelVBO *m : modelVBOs)
		delete m;
	delete missile;
}

/**
 * collisionShape[0] -> chassis
 * collisionShape[1-6] -> wheels
 * collisionShape[7] -> turret
 * collisionShape[8] -> canon
 *
 * rigidBody[0] -> chassis
 * rigidBody[1-6] -> wheels
 * rigidBody[7] -> turret
 * rigidBody[8] -> canon
 *
 * wheelHinge2[0-5] -> wheels
 *
 * modelVBOs[0] -> chassis
 * modelVBOs[1] -> wheels
 * modelVBOx[2] -> canon
 *
 */
void Tank::init(btDynamicsWorld *world, AAssetManager *mgr) {
	makeChassis(mgr);
	makeWheels(mgr, world);
	makeTurret(mgr, world);
	makeCanon(mgr, world);
}

void Tank::makeChassis(AAssetManager *mgr) {
	std::string chassisObjTxt = getFileText(mgr, "obj/tank_chassis.obj");

	btTransform tr;
	tr.setIdentity();

	btCollisionShape *chassisShape = parseObj(chassisObjTxt);
	collisionShape.push_back(chassisShape);

	tr.setOrigin(btVector3(spawnPos.x + 0, spawnPos.y + 0, spawnPos.z + 0));

	std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(chassisMass, tr,
																				 chassisShape);
	btRigidBody *m_carChassis = std::get<0>(tmp);
	btDefaultMotionState *m_carChassiMotionState = std::get<1>(tmp);

	modelVBOs.push_back(new ModelVBO(chassisObjTxt, chassisColor));
	scale.push_back(chassisScale);
	rigidBody.push_back(m_carChassis);
	defaultMotionState.push_back(m_carChassiMotionState);
}

void Tank::makeWheels(AAssetManager *mgr, btDynamicsWorld *world) {
	btTransform tr;

	btCollisionShape *m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));
	collisionShape.push_back(m_wheelShape);

	std::string cylObjText = getFileText(mgr, "obj/cylinderX.obj");
	scale.push_back(wheelScale);
	modelVBOs.push_back(new ModelVBO(cylObjText, wheelColor));

	for (int i = 0; i < nbWheel; i++) {

		btRigidBody *pBodyA = rigidBody[0];
		pBodyA->setActivationState(DISABLE_DEACTIVATION);

		tr.setIdentity();
		tr.setOrigin(wheelPos[i]);

		std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(wheelMass, tr, m_wheelShape);

		btRigidBody *pBodyB = std::get<0>(tmp);
		btDefaultMotionState *pBodyBMotionState = std::get<1>(tmp);

		rigidBody.push_back(pBodyB);
		defaultMotionState.push_back(pBodyBMotionState);

		pBodyB->setFriction(1110);
		pBodyB->setActivationState(DISABLE_DEACTIVATION);

		btVector3 parentAxis(0.f, 1.f, 0.f);
		btVector3 childAxis(1.f, 0.f, 0.f);
		btVector3 anchor = tr.getOrigin();
		wheelHinge2.push_back(new btHinge2Constraint(*pBodyA, *pBodyB, anchor, parentAxis,
												 childAxis));

		wheelHinge2[i]->setLowerLimit(float(-M_PI) * 0.5f);
		wheelHinge2[i]->setUpperLimit(float(M_PI) * 0.5f);

		world->addConstraint(wheelHinge2[i], true);

		/**
		 * Axis :
		 * 	- Linear, y = 2
		 * 	- Rotation, x = 3, z = 4, y = 5
		 */

		{
			int motorAxis = 3;
			wheelHinge2[i]->enableMotor(motorAxis, true);
			wheelHinge2[i]->setMaxMotorForce(motorAxis, 1e10f);
			wheelHinge2[i]->setTargetVelocity(motorAxis, 0);
		}

		{
			int motorAxis = 5;
			wheelHinge2[i]->setLimit(motorAxis, 0, 0);
		}

		{
			int index = 2;
			wheelHinge2[i]->setLimit(index, -0.4f, 0.f);
			wheelHinge2[i]->setDamping(index, 2.3f, true);
			wheelHinge2[i]->setStiffness(index, 20.f, true);
			wheelHinge2[i]->setBounce(index, 0.1f);
		}

		wheelHinge2[i]->setDbgDrawSize(btScalar(5.f));
	}

	{
		// Bug
		wheelHinge2[2]->setLimit(0, 0.f, 0.f);
		wheelHinge2[3]->setLimit(0, 0.f, 0.f);
		wheelHinge2[2]->setLimit(1, 0.f, 0.f);
		wheelHinge2[3]->setLimit(1, 0.f, 0.f);
		/*wheelHinge2[2]->setLimit(2, 0.f, 0.f);
		wheelHinge2[3]->setLimit(2, 0.f, 0.f);*/
	}
}

void Tank::makeTurret(AAssetManager *mgr, btDynamicsWorld *world) {
	btTransform tr;

	scale.push_back(turretScale);

	std::string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");

	modelVBOs.push_back(new ModelVBO(turretObjTxt, turretColor));

	btCollisionShape* turretShape = parseObj(turretObjTxt);
	turretShape->setLocalScaling(btVector3(scale[2].x, scale[2].y, scale[2].z));

	collisionShape.push_back(turretShape);

	tr.setIdentity();
	tr.setOrigin(btVector3(turretPos.x, turretPos.y, turretPos.z));

	std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(turretMass, tr, turretShape);

	btRigidBody *pBodyA = rigidBody[0];
	btRigidBody *pBodyB = std::get<0>(tmp);
	btDefaultMotionState *pBodyBMotionState = std::get<1>(tmp);

	rigidBody.push_back(pBodyB);
	defaultMotionState.push_back(pBodyBMotionState);

	btVector3 pivotA = btVector3(0.f, chassisScale.y, 0.f);
	btVector3 pivotB = btVector3(0.f, -turretScale.y, 0.f);
	btVector3 axis = btVector3(0.f, 1.f, 0.f);

	turretHinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
	world->addConstraint(turretHinge, true);
	turretHinge->setLimit(0,0);
}

void Tank::makeCanon(AAssetManager *mgr, btDynamicsWorld *world) {
	std::string canonObjTxt = getFileText(mgr, "obj/cylinderZ.obj");

	modelVBOs.push_back(new ModelVBO(canonObjTxt, turretColor));

	scale.push_back(canonScale);

	btCollisionShape* canonShape = new btCylinderShapeZ(btVector3(canonScale.x, canonScale.y, canonScale.z));

	collisionShape.push_back(canonShape);

	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(canonPos.x, canonPos.y, canonPos.z));

	std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(canonMass, tr, canonShape);

	btRigidBody *pBodyA = rigidBody[7];
	btRigidBody *pBodyB = std::get<0>(tmp);
	btDefaultMotionState *pBodyBMotionState = std::get<1>(tmp);

	rigidBody.push_back(pBodyB);
	defaultMotionState.push_back(pBodyBMotionState);

	tr.setIdentity();
	tr.setOrigin(btVector3(canonPos.x, canonPos.y, canonPos.z - canonScale.z));

	btVector3 pivotA = btVector3(0.f, 0.f, turretScale.z - canonOffset);
	btVector3 pivotB = btVector3(0.f, 0.f, -canonScale.z);
	btVector3 axis = btVector3(1,0,0);
	canonHinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
	world->addConstraint(canonHinge, true);
	//world->addRigidBody(pBodyB, 1, 0); // Canon subit pas de collision
	for (int i = 0; i < rigidBody.size() - 1; i++) {
		pBodyB->setIgnoreCollisionCheck(rigidBody[i], true);
	}
	canonHinge->setLimit(0,0);
}

void Tank::fire(vector<Base *> *bases) {
	if (!hasClickedShoot) {
		return;
	}
	hasClickedShoot = false;

	btScalar tmp[16];
	glm::vec3 missileScale = glm::vec3(0.1f, 0.3f, 0.1f);

	defaultMotionState[8]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	btQuaternion quat = defaultMotionState[8]->m_graphicsWorldTrans.getRotation();
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::vec4 vec = modelMatrix * glm::vec4(0.f, 0.f, canonScale.z + 1.f, 1.f);

	glm::mat4 rotMatrix = glm::toMat4(glm::quat(quat.getX(), quat.getY(), quat.getZ(), quat.getW()))
						  * glm::rotate(glm::mat4(1.f), 90.f, glm::vec3(1,0,0));

	Base* c = new Missile(missile, glm::vec3(vec.x, vec.y, vec.z), missileScale, rotMatrix, 10.f);
	world->addRigidBody(c->rigidBody[0]);

	glm::vec4 forceVec = modelMatrix * glm::vec4(0.f, 0.f, 500.f, 0.f);

	c->rigidBody[0]->applyCentralImpulse(btVector3(forceVec.x, forceVec.y, forceVec.z));

	bases->push_back(c);
}
