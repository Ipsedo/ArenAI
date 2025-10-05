//
// Created by samuel on 13/08/18.
//

#ifndef PHYVR_CHASSIS_H
#define PHYVR_CHASSIS_H

#include <android/asset_manager.h>

#include "../../controls.h"
#include "../../graphics/camera.h"
#include "../poly.h"
#include "glm/glm.hpp"

const glm::vec3 chassisScale = glm::vec3(1.2f, 0.5f, 2.f);
const float chassisMass = 10000.f;
const float chassisColor[4]{150.f / 255.f, 40.f / 255.f, 27.f / 255.f, 1.f};

class Chassis : public Controls, public Camera, public Poly {
private:
    bool respawn;
    btVector3 pos;
    bool isHit;

public:
    Chassis(AAssetManager *mgr, btVector3 pos);

    void onInput(input in) override;

    output getOutput() override;

    void update() override;

    void decreaseLife(int toSub) override;

    glm::vec3 camPos(bool VR) override;

    glm::vec3 camLookAtVec(bool VR) override;

    glm::vec3 camUpVec(bool VR) override;
};

#endif// PHYVR_CHASSIS_H
