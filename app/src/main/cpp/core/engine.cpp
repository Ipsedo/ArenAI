//
// Created by samuel on 26/05/18.
//
#include <android/log.h>
#include "engine.h"
#include "../entity/shooter.h"

static float gravity = -10.f;
static float deltaTime = 1.f / 60.f;

bool contact_callback(btManifoldPoint &btmanifoldpoint, const btCollisionObjectWrapper *btcollisionobject0, int part_0,
					  int index_0, const btCollisionObjectWrapper *btcollisionobject1, int part_1, int index_1) {

	/*btRigidBody* body0 = (btRigidBody*) btRigidBody::upcast(btcollisionobject0->getCollisionObject());
	btRigidBody* body1 = (btRigidBody*) btRigidBody::upcast(btcollisionobject1->getCollisionObject());

	btVector3 pos0 = body0->getWorldTransform().getOrigin();
	btVector3 pos1 = body1->getWorldTransform().getOrigin();

	body0->applyCentralForce((pos0 - pos1) * 2000);
	body1->applyCentralForce((pos0 - pos1) * 2000);*/

	__android_log_print(ANDROID_LOG_DEBUG, "AAA", "p");

	return false;
}

bool callback_finish(void *userPersistentData) {
	__android_log_print(ANDROID_LOG_DEBUG, "AAA", "yo");
	return false;
}

bool callback_processed(btManifoldPoint &cp, void *body0, void *body1) {
	__android_log_print(ANDROID_LOG_DEBUG, "AAA", "yo");
	return false;
}

Engine::Engine(vector<Base *> *bases) {

	this->bases = bases;

	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	broadPhase = new btDbvtBroadphase();
	constraintSolver = new btSequentialImpulseConstraintSolver();

	world = new btDiscreteDynamicsWorld(dispatcher,
										broadPhase,
										constraintSolver,
										collisionConfiguration);
	world->setGravity(btVector3(0, gravity, 0));

	for (Base *b : *this->bases)
		for (btRigidBody *bd : b->rigidBody)
			world->addRigidBody(bd);

	//gContactAddedCallback = contact_callback;
	//gContactDestroyedCallback = callback_finish;
	//gContactProcessedCallback = callback_processed;
}

void Engine::update(float delta) {
	// add rigid body
	for (Base* b : *bases) {
		for (btRigidBody *rb : b->rigidBody)
			if (!rb->isInWorld())
				world->addRigidBody(rb);
		b->update();
	}

	// remove base and rigidBody
	for (Base* b : *bases)
		if (b->isDead())
			for (btRigidBody* rb : b->rigidBody)
				world->removeRigidBody(rb);

	bases->erase(std::remove_if(bases->begin(), bases->end(),
								[](Base* o) { return o->isDead(); }),
				 bases->end());

	for (Shooter* s : shooters)
		s->fire(bases);

	world->stepSimulation(deltaTime);
}

void Engine::addShooter(Shooter *s) {
	shooters.push_back(s);
}

Engine::~Engine() {
	// From car bullet example
	for (int i = world->getNumCollisionObjects() - 1; i >= 0; i--) {
		btCollisionObject *obj = world->getCollisionObjectArray()[i];
		btRigidBody *body = btRigidBody::upcast(obj);
		if (body && body->getMotionState()) {

			while (body->getNumConstraintRefs()) {
				btTypedConstraint *constraint = body->getConstraintRef(0);
				world->removeConstraint(constraint);
				delete constraint;
			}
			//delete body->getMotionState(); base delete motionstate
			world->removeRigidBody(body);
		} else {
			world->removeCollisionObject(obj);
		}
		// delete obj; base delete body object (but btCollisionObject or btRigidBody ?)
	}

	delete world;
}
