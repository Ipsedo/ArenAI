//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include "../poly.h"
#include "../../controls.h"
#include "chassis.h"
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include "glm/glm.hpp"
#include <android/asset_manager.h>

#define MAX_FRAME_TOP_VEL 30.f

const float wheelRadius = 0.65f;
const float wheelWidth = 0.5f;

const float wheelOffset = 0.5f;

const float wheelbaseOffset = 0.1f;

const float wheelMass = 100.f;

const float wheelColor[4]{52.f / 255.f, 73.f / 255.f, 94.f / 255.f, 1.f};

const btVector3 wheelPos[8]{
		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, 0.75f * 3.f),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, 0.75f * 3.f),

		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, 0.75f),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, 0.75f),

		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, -0.75f),
		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, -0.75f),

		btVector3(+(chassisScale.x + wheelbaseOffset), -wheelOffset, -0.75f * 3.f),
		btVector3(-(chassisScale.x + wheelbaseOffset), -wheelOffset, -0.75f * 3.f)
};

class Wheel : public Poly, public Controls {
private:
	bool isBraking;
	bool isMotorEnabled;
	bool hasReAccelerate;
	bool freeWheel;

	float targetSpeed;
	bool respawn;
	btVector3 pos;
	btVector3 chassisPos;

	static ModelVBO *makeWheelMesh(AAssetManager *mgr);

protected:
	btGeneric6DofSpring2Constraint *hinge;

public:
	Wheel(AAssetManager *mgr,
		  btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos);

	/*
	 * Control overrides
	 */
	void onInput(input in) override;
	output getOutput() override;

	/*
	 * Base override
	 */
	void update() override;
};

class FrontWheel : public Wheel {
private:
	float direction;

public:
	FrontWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos,
			   btVector3 pos);

	void onInput(input in) override;

	void update() override;
};

#endif //PHYVR_WHEEL_H
