//
// Created by samuel on 26/05/18.
//
#include "engine.h"
#include "entity/ammu/explosion.h"
#include "entity/ammu/shooter.h"
#include "../utils/assets.h"
#include "../utils/rigidbody.h"
#include <algorithm>

static float gravity = -10.f;
static float deltaTime = 1.f / 60.f;

bool contact_callback(btManifoldPoint &btmanifoldpoint, const btCollisionObjectWrapper *btcollisionobject0, int part_0,
					  int index_0, const btCollisionObjectWrapper *btcollisionobject1, int part_1, int index_1) {

	btRigidBody *body0 = (btRigidBody *) btRigidBody::upcast(btcollisionobject0->getCollisionObject());
	btRigidBody *body1 = (btRigidBody *) btRigidBody::upcast(btcollisionobject1->getCollisionObject());

	Base *downcast0 = static_cast<Base *>(body0);
	Base *downcast1 = static_cast<Base *>(body1);

	downcast0->decreaseLife(1);
	downcast1->decreaseLife(1);

	return false;
}

bool callback_finish(void *userPersistentData) {
	tuple<Base *, Base *> *t = (tuple<Base *, Base *> *) userPersistentData;

	Base *b0 = get<0>(*t);
	Base *b1 = get<1>(*t);
	b0->onContactFinish(b1);
	b1->onContactFinish(b0);

	b0->decreaseLife(1);
	b1->decreaseLife(1);

	delete t;

	return false;
}

bool callback_processed(btManifoldPoint &cp, void *body0, void *body1) {
	Base *b0 = (Base *) body0;
	Base *b1 = (Base *) body1;

	cp.m_userPersistentData = new tuple<Base *, Base *>(b0, b1);

	return false;
}

Engine::Engine(Level *level, AAssetManager *mgr)
		: level(level),
		  explosion(new TransparentModelVBO(getFileText(mgr, "obj/sphere.obj"), new float[4]{1.f, 0.6f, 0.f, 0.7f})) {

	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	broadPhase = new btDbvtBroadphase();
	constraintSolver = new btSequentialImpulseConstraintSolver();

	world = new btDiscreteDynamicsWorld(dispatcher,
										broadPhase,
										constraintSolver,
										collisionConfiguration);
	world->setGravity(btVector3(0, gravity, 0));

	//gContactAddedCallback = contact_callback;
	gContactDestroyedCallback = callback_finish;
	gContactProcessedCallback = callback_processed;
}

void Engine::update(float delta) {
	// add rigid body
	for (Base *b : level->getEntities()) {
		if (!b->isInWorld())
			world->addRigidBody(b);
		b->update();
	}

	for (Shooter *s : level->getShooters())
		level->addBases(s->fire());

	world->stepSimulation(deltaTime);

	limits = level->getLimits();

	// remove base and rigidBody
	vector<Base *> toAdd;
	level->deleteBase([this, &toAdd](Base *b) {
		bool isDead = b->isDead() || !limits->isInside(b);
		if (isDead) {
			if (b->needExplosion()) {
				Explosion *e = new Explosion(b->getWorldTransform().getOrigin(), explosion);
				toAdd.push_back(e);
			}
			deleteBase(b);
		}
		return isDead;
	});
	level->addBases(toAdd);
}

Engine::~Engine() {
	// From car bullet example
	for (int i = world->getNumCollisionObjects() - 1; i >= 0; i--) {
		btCollisionObject *obj = world->getCollisionObjectArray()[i];
		Base *base = (Base *) btRigidBody::upcast(obj);
		if (base && base->getMotionState()) {
			while (base->getNumConstraintRefs()) {
				btTypedConstraint *constraint = base->getConstraintRef(0);
				world->removeConstraint(constraint);
				delete constraint;
			}
			delete base->getMotionState();
			delete base->getCollisionShape();
			world->removeRigidBody(base);
		} else {
			world->removeCollisionObject(obj);
			delete obj;
		}
		delete base;
	}

	delete world;
	delete broadPhase;
	delete dispatcher;
	delete collisionConfiguration;
	delete constraintSolver;
	delete limits;
}

void Engine::deleteBase(Base *base) {
	if (base && base->getMotionState()) {
		while (base->getNumConstraintRefs()) {
			btTypedConstraint *constraint = base->getConstraintRef(0);
			world->removeConstraint(constraint);
			delete constraint;
		}
		delete base->getMotionState();
		delete base->getCollisionShape();
		world->removeRigidBody(base);
	}
	delete base;
}