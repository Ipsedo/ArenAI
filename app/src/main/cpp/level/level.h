//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H


#include "../graphics/renderer.h"
#include "../entity/box.h"

class Level {
public:
    Level(vector<Box*>* boxes);
    void update(float delta);
    // ajoute un nouveau Box SEULEMENT au World (sera suppr dans le futur)
    void addNewBox(Box* box);
    ~Level();

private:
    vector<Box*>* boxes;
    btDiscreteDynamicsWorld* world;
    btBroadphaseInterface* broadPhase;
    btCollisionDispatcher* dispatcher;
    btDefaultCollisionConfiguration* collisionConfiguration;
    btSequentialImpulseConstraintSolver* constraintSolver;
};


#endif //PHYVR_LEVEL_H
