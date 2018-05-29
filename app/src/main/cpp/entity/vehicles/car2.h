//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_CAR2_H
#define PHYVR_CAR2_H


#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../base.h"

class Car2 : public Base {
public:
    Car2(btDynamicsWorld* world, AAssetManager* mgr);
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;
    void control();
    ~Car2();
private:
    btDefaultVehicleRaycaster* raycaster;
    btRaycastVehicle::btVehicleTuning tuning;
    btRaycastVehicle* vehicle;
    void init(btDynamicsWorld* world, AAssetManager* mgr);
    ModelVBO* chassis;
    ModelVBO* wheel;
};


#endif //PHYVR_CAR2_H
