//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_TANK2_H
#define PHYVR_TANK2_H


#include "entity/base.h"
#include <android/asset_manager.h>
#include <controls/controls.h>
#include "../../../graphics/camera.h"
#include "wheel.h"
#include "turret.h"

class Tank {
private:
	Chassis *chassis;
	Turret *turret;
	Canon *canon;
	vector<Wheel *> wheels;
	Camera *camera;
public:
	Tank(bool vr, AAssetManager *mgr, btDynamicsWorld *world, btVector3 centerPos);

	vector<Base *> getBaseTest();

	vector<Controls *> getControls();

	Camera *getCamera();

	vector<Shooter *> getShooters();

};


#endif //PHYVR_TANK2_H
