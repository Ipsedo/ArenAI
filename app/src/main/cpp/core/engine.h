//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_ENGINE_H
#define PHYVR_ENGINE_H

#include "../levels/level.h"
#include "../graphics/renderer.h"
#include "../entity/base.h"
#include "entity/ammu/shooter.h"
#include "limits.h"

bool contact_callback(btManifoldPoint &btmanifoldpoint,
					  const btCollisionObjectWrapper *btcollisionobject0,
					  int part_0, int index_0,
					  const btCollisionObjectWrapper *btcollisionobject1,
					  int part_1, int index_1);

bool callback_finish(void *userPersistentData);

bool callback_processed(btManifoldPoint &cp, void *body0, void *body1);

class Engine {
public:
	Engine(Level *level, AAssetManager *mgr);

	void update(float delta);

	~Engine();

	btDiscreteDynamicsWorld *world;

private:
	Level *level;
	btBroadphaseInterface *broadPhase;
	btCollisionDispatcher *dispatcher;
	btDefaultCollisionConfiguration *collisionConfiguration;
	btSequentialImpulseConstraintSolver *constraintSolver;
	DiffuseModel *explosion;

	void deleteBase(Base *base);
};


#endif //PHYVR_LEVEL_H
