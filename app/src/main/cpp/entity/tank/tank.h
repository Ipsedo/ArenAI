//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_TANK2_H
#define PHYVR_TANK2_H


#include "../player.h"
#include <android/asset_manager.h>
#include "../../graphics/camera.h"
#include "wheel.h"
#include "turret.h"
#include "canon.h"
#include "../../ui/solid_curve.h"

class Tank : public Player {
private:
	Chassis *chassis;
	Turret *turret;
	Canon *canon;
	vector<Wheel *> wheels;
	Camera *camera;
	Curve *curve;

public:
	Tank(bool vr, AAssetManager *mgr, btDynamicsWorld *world, btVector3 centerPos);

	vector<Base *> getBases() override;

	vector<Controls *> getControls() override;

	Camera *getCamera() override;

	vector<Shooter *> getShooters() override;

	vector<Drawable *> getDrawables() override;

};


#endif //PHYVR_TANK2_H
