//
// Created by samuel on 13/08/18.
//

#ifndef PHYVR_TURRET_H
#define PHYVR_TURRET_H

#include "../poly.h"
#include <glm/glm.hpp>
#include "../../controls.h"
#include "../ammu/shooter.h"
#include "chassis.h"

const glm::vec3 turretScale(0.9f, 0.25f, 1.2f);
const btVector3 turretRelPos(0.f, chassisScale.y + turretScale.y, 0.f);
const float turretColor[4]{4.f / 255.f, 147.f / 255.f, 114.f / 255.f, 1.f};
const float turretMass = 100.f;

class Turret : public Controls, public Poly {
private:
	// Controls
	float angle;

	bool respawn;

	float added;

	bool isHit;

	// Turret
	btVector3 pos;

	btHingeConstraint *hinge;
public:
	Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos);

	/*
	 * Control overrides
	 */
	void onInput(input in) override;
	output getOutput() override;

	/*
	 * Base override
	 */
	void update() override;

	void decreaseLife(int toSub) override;
};

#endif //PHYVR_TURRET_H
