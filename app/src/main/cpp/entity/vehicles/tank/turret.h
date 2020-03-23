//
// Created by samuel on 13/08/18.
//

#ifndef PHYVR_TURRET_H
#define PHYVR_TURRET_H


#include "../../poly.h"
#include <glm/glm.hpp>
#include "../../../controls/controls.h"
#include "../../ammu/shooter.h"
#include "chassis.h"

static const glm::vec3 turretScale(0.9f, 0.25f, 1.2f);
static const btVector3 turretRelPos(0.f, chassisScale.y + turretScale.y, 0.f);
static float turretColor[4]{4.f / 255.f, 147.f / 255.f, 114.f / 255.f, 1.f};
static const float turretMass = 100.f;

class Turret : public Controls, public Poly {
private:
	// Controls
	float angle;

	bool respawn;

	float added;

	// Turret
	btVector3 pos;

	btHingeConstraint *hinge;
public:
	Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos);

	void onInput(input in) override;

	void update() override;

};

#endif //PHYVR_TURRET_H
