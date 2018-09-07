//
// Created by samuel on 29/08/18.
//

#ifndef PHYVR_LEVEL1_H
#define PHYVR_LEVEL1_H

#include "../level.h"
#include "cible.h"
#include "../../entity/vehicles/tank/tank.h"
#include "../../entity/ground/map.h"
#include "../../graphics/drawable/cubemap.h"
#include "../../ui/compass.h"

/**
 * Niveau 0 (training)
 * idée niveau :
 * 1) se rendre à un point précis
 * 2) tirer sur des cibles
 * 	  les cibles sont montées sur charnière et basculent
 */
class Level1 : public Level {

public:
	Level1();

	void init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) override;

	vector<Controls *> getControls() override;

	Camera *getCamera() override;

	vector<Shooter *> getShooters() override;

	vector<Base *> getEntities() override;

	vector<Drawable *> getDrawables() override;

	Limits getLimits() override;

	bool won() override;

	bool lose() override;

	void step() override ;

	~Level1() override;

private:
	bool isInit;

	Tank *tank;

	vector<Cible *> cibles;

	vector<Compass *> compass;

	CubeMap *map;
};


#endif //PHYVR_LEVEL0_H
