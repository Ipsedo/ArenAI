//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include "entity/base.h"
#include "../../../controls/controls.h"
#include "chassis.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>

static float wheelRadius = 0.8f;
static float wheelWidth = 0.4f;

static float wheelOffset = 0.6f;

static float wheelbaseOffset = 0.1f;

static float wheelMass = 300.f;

static float wheelColor[4]{52.f / 255.f, 73.f / 255.f, 94.f / 255.f, 1.f};

static btVector3 wheelPos[6]{
		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, +1.75f),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, +1.75f),
		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, +0),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, +0),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, -1.75f),
		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, -1.75f)
};

class Wheel : public Base, public Controls {
private:
	bool isBraking;
	bool isMotorEnabled;
	bool hasReAccelerate;
	float targetSpeed;
	bool respawn;
	btVector3 pos;
	btVector3 chassisPos;

protected:
	btGeneric6DofSpring2Constraint *hinge;

public:
	/**
	 *
	 * @param constructionInfo
	 * @param modelVBO
	 * @param scale
	 * @param chassis
	 * @param pos Position relative de la roue par rapport au centre du chassis
	 */
	Wheel(const btRigidBodyConstructionInfo &constructionInfo,
		  ModelVBO *modelVBO, const glm::vec3 &scale,
		  btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos);

	void onInput(input in) override;

	void update() override;
};

class FrontWheel : public Wheel {
private:
	float direction;

public:
	FrontWheel(const btRigidBodyConstructionInfo &constructionInfo,
			   ModelVBO *modelVBO, const glm::vec3 &scale,
			   btDynamicsWorld *world, Base *chassis, const btVector3 &chassisPos, const btVector3 &pos);

	void onInput(input in) override;

	void update() override;
};


Wheel *makeWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos);

FrontWheel *makeFrontWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos);

#endif //PHYVR_WHEEL_H
