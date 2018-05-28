//
// Created by samuel on 26/05/18.
//

#include "level.h"

Level::Level(vector<Base*>* bases) {

    this->bases = bases;

    collisionConfiguration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collisionConfiguration);
    broadPhase = new btDbvtBroadphase();
    constraintSolver = new btSequentialImpulseConstraintSolver();

    world = new btDiscreteDynamicsWorld(dispatcher,
                                        broadPhase,
                                        constraintSolver,
                                        collisionConfiguration);
    world->setGravity(btVector3(0,-10,0));
    for (Base* b : *this->bases)
        for (btRigidBody* bd : b->rigidBody)
            world->addRigidBody(bd);
}

void Level::update(float delta) {
    world->stepSimulation(delta);
}

void Level::addNewBox(Base *b) {
    for (btRigidBody* bd : b->rigidBody)
        world->addRigidBody(bd);
}

Level::~Level() {
    /*delete collisionConfiguration;
    delete dispatcher;

    delete constraintSolver;
    delete broadPhase;*/

    delete world;
}
