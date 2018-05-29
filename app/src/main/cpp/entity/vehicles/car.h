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
    void control();
    ~Car();
private:
    void init(btDynamicsWorld* world, AAssetManager* mgr);
    /*btVector3 m_loadStartPos;
    btVector3	m_forkStartPos;
    btHingeConstraint* m_liftHinge;
    btSliderConstraint* m_forkSlider;*/
    btVehicleRaycaster m_vehicleRayCaster;
    btRaycastVehicle m_vehicle;
    std::vector<ModelVBO*> modelVBOs;
};


#endif //PHYVR_CAR_H
