//
// Created by samuel on 13/08/18.
//

#ifndef PHYVR_TURRET_H
#define PHYVR_TURRET_H


#include "../../poly/poly.h"
#include <glm/glm.hpp>
#include "../../../controls/controls.h"
#include "../../shooter.h"
#include "chassis.h"

static const glm::vec3 turretScale(0.9f, 0.25f, 1.2f);
static const btVector3 turretRelPos(0.f, chassisScale.x + turretScale.y, 0.f);
static float turretColor[4]{4.f / 255.f, 147.f / 255.f, 114.f / 255.f, 1.f};
static const float turretMass = 100.f;

class Turret : public Poly, public Controls {
private:
	btHingeConstraint *hinge;
	float angle;
	bool respawn;
	btVector3 pos;
public:
	Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos);

	void onInput(input in) override;

	void update() override;

};

static float canonMass = 10.f;
static float canonOffset = 0.1f;
static const glm::vec3 canonScale(0.1f, 0.1f, 0.8f);
static const btVector3 canonRelPos(0.f, 0.f, turretScale.z + canonScale.z - canonOffset);

class Canon : public Poly, public Controls, public Shooter, public Camera {
private:
	float angle;
	bool respawn;
	bool hasClickedShoot;
	btVector3 pos;
	btHingeConstraint *hinge;
	DiffuseModel *missile;
public:
	Canon(AAssetManager *mgr, btDynamicsWorld *world, Base *turret, btVector3 turretPos);

	void onInput(input in) override;

	void update() override;

	void fire(std::vector<Base *> *entities) override;

	glm::vec3 camPos(bool VR) override;

	glm::vec3 camLookAtVec(bool VR) override;

	glm::vec3 camUpVec(bool VR) override;
};

#endif //PHYVR_TURRET_H
