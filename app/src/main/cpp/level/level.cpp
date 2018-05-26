//
// Created by samuel on 26/05/18.
//

#include "level.h"

Level::Level(vector<Box*>* boxes) {

    this->boxes = boxes;

    collisionConfiguration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collisionConfiguration);
    broadPhase = new btDbvtBroadphase();
    constraintSolver = new btSequentialImpulseConstraintSolver();

    world = new btDiscreteDynamicsWorld(dispatcher,
                                        broadPhase,
                                        constraintSolver,
                                        collisionConfiguration);
    world->setGravity(btVector3(0,-10,0));
    for (Box* b : *this->boxes)
        world->addRigidBody(b->rigidBody);
}

void Level::update(float delta) {
    world->stepSimulation(delta);
}

void Level::addNewBox(Box *b) {
    world->addRigidBody(b->rigidBody);
}

Level::~Level() {
    /*delete collisionConfiguration;
    delete dispatcher;

    delete constraintSolver;
    delete broadPhase;*/

    delete world;
}
