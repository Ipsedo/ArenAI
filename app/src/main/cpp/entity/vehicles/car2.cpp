//
// Created by samuel on 29/05/18.
//

#include "car2.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

static int rightIndex = 0;
static int upIndex = 1;
static int forwardIndex = 2;

static float wheelRadius = 0.5f;
static float wheelWidth = 0.2f;

#define CUBE_HALF_EXTENTS 1.f

Car2::Car2(btDynamicsWorld *world, AAssetManager *mgr) {
	init(world, mgr);

	int wheelIndex = 2;
	vehicle->applyEngineForce(10, wheelIndex);
	wheelIndex = 3;
	vehicle->applyEngineForce(10, wheelIndex);

}

void Car2::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	btScalar tmp[16];
	btTransform tr = vehicle->getChassisWorldTransform();
	tr.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[0]);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	chassis->draw(mvpMatrix, mvMatrix, lighPos);

	for (int i = 0; i < vehicle->getNumWheels(); i++) {
		vehicle->updateWheelTransform(i, true);
		vehicle->getWheelInfo(i).m_worldTransform.getOpenGLMatrix(tmp);
		modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[1]);

		mvMatrix = vMatrix * modelMatrix;
		mvpMatrix = pMatrix * mvMatrix;

		wheel->draw(mvpMatrix, mvMatrix, lighPos);
	}
}

void Car2::control() {

}

Car2::~Car2() {

}

/**
 * https://github.com/bulletphysics/bullet3/pull/448/files?diff=split
 * @param world
 * @param mgr
 */
void Car2::init(btDynamicsWorld *world, AAssetManager *mgr) {

	std::string cubeObjTxt = getFileText(mgr, "obj/cube.obj");
	chassis = new ModelVBO(cubeObjTxt,
						   new float[4]{float(rand()) / RAND_MAX,
										float(rand()) / RAND_MAX,
										float(rand()) / RAND_MAX,
										1.f}
	);

	// Chassis rigidBody
	btTransform tr;
	tr.setIdentity();

	btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
	collisionShape.push_back(chassisShape);

	tr.setOrigin(btVector3(0, 5.f, 0));

	std::tuple<btRigidBody *, btDefaultMotionState *> tmp = localCreateRigidBody(800, tr, chassisShape);//chassisShape);
	btRigidBody *m_carChassis = std::get<0>(tmp);
	btDefaultMotionState *m_carChassiMotionState = std::get<1>(tmp);
	world->addRigidBody(m_carChassis);

	scale.push_back(glm::vec3(1.f, 0.5f, 2.f));
	rigidBody.push_back(m_carChassis);
	defaultMotionState.push_back(m_carChassiMotionState);


	// vehicle
	raycaster = new btDefaultVehicleRaycaster(world);
	vehicle = new btRaycastVehicle(tuning, m_carChassis, raycaster);

	///never deactivate the vehicle
	m_carChassis->setActivationState(DISABLE_DEACTIVATION);

	world->addVehicle(vehicle);

	float connectionHeight = 1.2f;

	bool isFrontWheel = true;

	//choose coordinate system
	vehicle->setCoordinateSystem(rightIndex, upIndex, forwardIndex);

	btVector3 wheelDirectionCS0(0, -1, 0);
	btVector3 wheelAxleCS(-1, 0, 0);
	btScalar suspensionRestLength(0.6);

	std::string cylObjText = getFileText(mgr, "obj/cylinderX.obj");
	scale.push_back(glm::vec3(wheelWidth, wheelRadius, wheelRadius));
	wheel = new ModelVBO(cylObjText, new float[4]{0.f, 1.f, 0.f, 1.f});

	btVector3 connectionPointCS0(CUBE_HALF_EXTENTS - (0.3f * wheelWidth), connectionHeight,
								 2 * CUBE_HALF_EXTENTS - wheelRadius);
	vehicle->addWheel(connectionPointCS0, wheelDirectionCS0, wheelAxleCS, suspensionRestLength, wheelRadius, tuning,
					  isFrontWheel);

	connectionPointCS0 = btVector3(-CUBE_HALF_EXTENTS + (0.3f * wheelWidth), connectionHeight,
								   2 * CUBE_HALF_EXTENTS - wheelRadius);
	vehicle->addWheel(connectionPointCS0, wheelDirectionCS0, wheelAxleCS, suspensionRestLength, wheelRadius, tuning,
					  isFrontWheel);

	isFrontWheel = false;
	connectionPointCS0 = btVector3(-CUBE_HALF_EXTENTS + (0.3f * wheelWidth), connectionHeight,
								   -2 * CUBE_HALF_EXTENTS + wheelRadius);
	vehicle->addWheel(connectionPointCS0, wheelDirectionCS0, wheelAxleCS, suspensionRestLength, wheelRadius, tuning,
					  isFrontWheel);

	connectionPointCS0 = btVector3(CUBE_HALF_EXTENTS - (0.3f * wheelWidth), connectionHeight,
								   -2 * CUBE_HALF_EXTENTS + wheelRadius);
	vehicle->addWheel(connectionPointCS0, wheelDirectionCS0, wheelAxleCS, suspensionRestLength, wheelRadius, tuning,
					  isFrontWheel);

	for (int i = 0; i < vehicle->getNumWheels(); i++) {
		btWheelInfo &wheel = vehicle->getWheelInfo(i);
		wheel.m_suspensionStiffness = 20.f;
		wheel.m_wheelsDampingRelaxation = 2.3f;
		wheel.m_wheelsDampingCompression = 4.4f;
		wheel.m_frictionSlip = 1000.f;
		wheel.m_rollInfluence = 0.1f;
	}
}
