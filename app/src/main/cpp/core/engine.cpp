//
// Created by samuel on 26/05/18.
//
#include "engine.h"
#include "../entity/shooter.h"

static float gravity = -10.f;
static float deltaTime = 1.f / 60.f;

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
}

void Engine::update(float delta) {
	// add rigid body
	for (Base* b : *bases) {
		for (btRigidBody *rb : b->rigidBody)
			if (!rb->isInWorld())
				world->addRigidBody(rb);
		b->update();
	}

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
