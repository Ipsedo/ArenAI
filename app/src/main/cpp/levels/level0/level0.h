//
// Created by samuel on 11/06/18.
//

#ifndef PHYVR_LEVEL0_H
#define PHYVR_LEVEL0_H

#include "../level.h"
#include "../../entity/vehicles/tank.h"

class Level0 : public Level {
public:
	Level0();

	void init() override;

	Controls* getControls() override;
	vector<Shooter *> getShooters() override;
	Camera *getCamera() override;

	vector<Base*> getEntities() override;

	bool won() override;
	bool lose() override;

private:
	Tank* tank;
	vector<Base*> entities;
	bool isInit;
};


#endif //PHYVR_LEVEL0_H
