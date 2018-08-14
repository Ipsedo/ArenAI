//
// Created by samuel on 26/05/18.
//
#include "engine.h"
#include "../entity/shooter.h"
#include "../utils/rigidbody.h"

static float gravity = -10.f;
static float deltaTime = 1.f / 60.f;

bool contact_callback(btManifoldPoint &btmanifoldpoint, const btCollisionObjectWrapper *btcollisionobject0, int part_0,
					  int index_0, const btCollisionObjectWrapper *btcollisionobject1, int part_1, int index_1) {

	/*btRigidBody* body0 = (btRigidBody*) btRigidBody::upcast(btcollisionobject0->getCollisionObject());
	btRigidBody* body1 = (btRigidBody*) btRigidBody::upcast(btcollisionobject1->getCollisionObject());

	btRigidBodyWithBase* downcast0 = static_cast<btRigidBodyWithBase*>(body0);
	btRigidBodyWithBase* downcast1 = static_cast<btRigidBodyWithBase*>(body1);

	downcast0->base->decreaseLife(1);
	downcast1->base->decreaseLife(1);*/

	return false;
}

bool callback_finish(void *userPersistentData) {
	/*std::tuple<Base*, Base*>* t = (std::tuple<Base*, Base*>*) userPersistentData;

	Base* b0 = std::get<0>(*t);
	Base* b1 = std::get<1>(*t);

	b0->decreaseLife(1);
	b1->decreaseLife(1);

	delete t;*/

	return false;
}

bool callback_processed(btManifoldPoint &cp, void *body0, void *body1) {
	/*btRigidBodyWithBase* b0 = (btRigidBodyWithBase*) body0;
	btRigidBodyWithBase* b1 = (btRigidBodyWithBase*) body1;

	cp.m_userPersistentData = new std::tuple<Base*, Base*>(b0->base, b1->base);*/

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
		world->addRigidBody(b);

	//gContactAddedCallback = contact_callback;
	gContactDestroyedCallback = callback_finish;
	gContactProcessedCallback = callback_processed;
}

void Engine::update(float delta) {
	// add rigid body
	for (Base *b : *bases) {
		if (!b->isInWorld())
			world->addRigidBody(b);
		b->update();
	}

	// remove base and rigidBody
	for (Base *b : *bases)
		if (b->isDead())
			world->removeRigidBody(b);

	bases->erase(std::remove_if(bases->begin(), bases->end(),
								[](Base *o) { return o->isDead(); }),
				 bases->end());

	for (Shooter *s : shooters)
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
