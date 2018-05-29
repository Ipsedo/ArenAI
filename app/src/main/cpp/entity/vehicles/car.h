//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CAR_H
#define PHYVR_CAR_H


#include "../base.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>

class Car : public Base {
public:
    Car(btDynamicsWorld* world, AAssetManager* mgr);
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;
    glm::vec3 getCamPos();
    void control();
    ~Car();
private:
    void init(btDynamicsWorld* world, AAssetManager* mgr);
    std::vector<btHinge2Constraint*> pHinge2;
    std::vector<ModelVBO*> modelVBOs;
};


#endif //PHYVR_CAR_H
