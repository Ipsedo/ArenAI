//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H

#include "../entity/base.h"
#include "../controls/controls.h"
#include "../graphics/camera.h"
#include "../entity/shooter.h"

class Level {
public:
	virtual void init() = 0;

	virtual Controls* getControls() = 0;
	virtual Camera* getCamera() = 0;
	virtual vector<Shooter*> getShooters() = 0;
	virtual vector<Base*> getEntities() = 0;

	virtual bool won() = 0;
	virtual bool lose() = 0;
};


#endif //PHYVR_LEVEL_H
