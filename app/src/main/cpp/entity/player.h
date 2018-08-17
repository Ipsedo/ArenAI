//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_PLAYER_H
#define PHYVR_PLAYER_H

#include "base.h"
#include "../controls/controls.h"
#include "shooter.h"
#include "../graphics/camera.h"

class Player {
public:
	virtual vector<Base *> getBase() = 0;

	virtual vector<Controls *> getControls() = 0;

	virtual Camera *getCamera() = 0;

	virtual vector<Shooter *> getShooters() = 0;
};

#endif //PHYVR_PLAYER_H
