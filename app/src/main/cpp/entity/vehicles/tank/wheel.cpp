//
// Created by samuel on 26/06/18.
//

#include "../../../utils/assets.h"
#include "wheel.h"
#include <glm/vec3.hpp>
#include "../../../utils/rigidbody.h"

Wheel::Wheel(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
			 ModelVBO *modelVBO, const glm::vec3 &scale,
			 btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 wheelPos) :
		Base(constructionInfo, modelVBO, scale),
		pos(wheelPos), chassisPos(chassisPos),
		isMotorEnabled(false), isBraking(true), targetSpeed(0.f), hasReAccelerate(false) {

	setFriction(500);
	setActivationState(DISABLE_DEACTIVATION);

	btTransform trA, trB;
	trA.setIdentity();
	trA.setOrigin(pos);
	trB.setIdentity();
	trB.setOrigin(btVector3(0, 0, 0));
	hinge = new btGeneric6DofSpring2Constraint(*chassis, *this, trA, trB, RO_XYZ);

	// Angular limits
	hinge->setAngularLowerLimit(btVector3(1, 0, 0));
	hinge->setAngularUpperLimit(btVector3(-1, 0, 0));

	// Linear limits
	hinge->setLinearLowerLimit(btVector3(0, -0.4f, 0));
	hinge->setLinearUpperLimit(btVector3(0, 0, 0));

	world->addConstraint(hinge, true);

	{
		int motorAxis = 3;
		hinge->enableMotor(motorAxis, isMotorEnabled);
		hinge->setMaxMotorForce(motorAxis, 1e10f);
		hinge->setTargetVelocity(motorAxis, targetSpeed);
	}

	{
		int index = 1;
		hinge->setDamping(index, 2.3f, true);
		hinge->setStiffness(index, 20.f, true);
		hinge->setBounce(index, 0.1f);
	}

	hinge->setDbgDrawSize(btScalar(5.f));

	//world->addRigidBody(this);
}


void Wheel::onInput(input in) {
	isMotorEnabled = abs(in.speed) > 1e-2f;
	targetSpeed = isMotorEnabled ? in.speed : 0.f;
	respawn = in.respawn;
	if (isMotorEnabled) hasReAccelerate = true;
	if (hasReAccelerate) isBraking = in.brake;
	if (isBraking) hasReAccelerate = false;
}

void Wheel::update() {
	Base::update();

	if (isBraking) {
		isMotorEnabled = true;
		targetSpeed = 0.f;
	}

	int motorAxis = 3;
	hinge->enableMotor(motorAxis, isMotorEnabled);
	hinge->setTargetVelocity(motorAxis, -targetSpeed * 10.f);

	if (respawn) {
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(pos + chassisPos);

		getMotionState()->setWorldTransform(tr);
		setWorldTransform(tr);
		clearForces();
		setLinearVelocity(btVector3(0, 0, 0));
		setAngularVelocity(btVector3(0, 0, 0));
		respawn = false;
	}
}


/**
 * Front wheels
 * Direction en plus pour Controls
 *
 *
 */
FrontWheel::FrontWheel(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
					   ModelVBO *modelVBO, const glm::vec3 &scale, btDynamicsWorld *world,
					   Base *chassis, const btVector3 &chassisPos, const btVector3 &pos)
		: Wheel(constructionInfo, modelVBO, scale, world, chassis, chassisPos, pos), direction(0.f) {}

void FrontWheel::onInput(input in) {
	Wheel::onInput(in);
	direction = in.xAxis;
}

void FrontWheel::update() {
	Wheel::update();

	int motorAxis = 4;
	hinge->setLimit(
			motorAxis,
			float(M_PI) * direction / 10.f,
			float(M_PI) * direction / 10.f);
}

ModelVBO *makeWheelMesh(AAssetManager *mgr) {
	std::string cylObjText = getFileText(mgr, "obj/cylinderX.obj");
	return new ModelVBO(cylObjText, wheelColor);
}

Wheel *makeWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos) {
	btCollisionShape *m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(pos + chassisPos);
	btRigidBody::btRigidBodyConstructionInfo cinfo
			= localCreateInfo(wheelMass, tr, m_wheelShape);
	return new Wheel(cinfo, makeWheelMesh(mgr),
					 glm::vec3(wheelWidth, wheelRadius, wheelRadius), world,
					 chassis, chassisPos, pos);
}

FrontWheel *makeFrontWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos) {
	btCollisionShape *m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(pos + chassisPos);
	btRigidBody::btRigidBodyConstructionInfo cinfo = localCreateInfo(wheelMass, tr, m_wheelShape);
	return new FrontWheel(cinfo, makeWheelMesh(mgr),
						  glm::vec3(wheelWidth, wheelRadius, wheelRadius), world, chassis,
						  chassisPos, pos);
}
