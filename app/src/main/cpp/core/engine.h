//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H

#include "../graphics/renderer.h"
#include "entity/base.h"
#include "../entity/shooter.h"

bool contact_callback(btManifoldPoint &btmanifoldpoint,
					  const btCollisionObjectWrapper *btcollisionobject0,
					  int part_0, int index_0,
					  const btCollisionObjectWrapper *btcollisionobject1,
					  int part_1, int index_1);

bool callback_finish(void *userPersistentData);

bool callback_processed(btManifoldPoint &cp, void *body0, void *body1);

class Engine {
public:
	Engine(vector<Base *> *b);

	void update(float delta);

	void addShooter(Shooter *s);

	~Engine();

	btDiscreteDynamicsWorld *world;

private:
	vector<Shooter *> shooters;
	vector<Base *> *bases;
	btBroadphaseInterface *broadPhase;
	btCollisionDispatcher *dispatcher;
	btDefaultCollisionConfiguration *collisionConfiguration;
	btSequentialImpulseConstraintSolver *constraintSolver;
};


#endif //PHYVR_LEVEL_H
