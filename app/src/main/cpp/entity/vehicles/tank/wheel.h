//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include "../../poly.h"
#include "../../../controls/controls.h"
#include "chassis.h"
#include "../../../utils/rigidbody.h"
#include "../../../utils/assets.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>

#define MAX_FRAME_TOP_VEL 30.f

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

class Wheel : public Poly, public Controls {
private:
	bool isBraking;
	bool isMotorEnabled;
	bool hasReAccelerate;
	int nbFrameMotorEnabled;
	float added;
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

	void onInput(input in) override;

	void update() override;
};

class FrontWheel : public Wheel {
private:
	float direction;

public:
	FrontWheel(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos, btVector3 pos);

	void onInput(input in) override;

	void update() override;
};

#endif //PHYVR_WHEEL_H
