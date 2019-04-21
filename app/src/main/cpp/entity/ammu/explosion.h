//
// Created by samuel on 20/08/18.
//

#ifndef PHYVR_EXPLOSION_H
#define PHYVR_EXPLOSION_H

#include <BulletCollision/CollisionDispatch/btGhostObject.h>
#include "entity/simple/sphere.h"
#include "../simple/tetra.h"


class Particules : public Tetra {
private:
	int nbFrames;
public:
	Particules(btVector3 explosionCenter, DiffuseModel *tetraede);

	bool isDead() override;

	void update() override;

	bool needExplosion() override;
};


class Explosion : public Poly {
private:
	int nbFrames;
public:
	Explosion(btVector3 pos, DiffuseModel *modelVBO);

	bool isDead() override;

	void update() override;

	bool needExplosion() override;

	void onContactFinish(Base *other) override;
};


#endif //PHYVR_EXPLOSION_H
