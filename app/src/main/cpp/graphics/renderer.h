//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <vector>
#include <glm/glm.hpp>
#include <android/asset_manager.h>
#include "../entity/vehicles/car.h"

#include "drawable/modelvbo.h"

class Renderer {
public:
    Renderer(vector<Base*>* bases);
    void update(glm::mat4 mHeadView);
    void setCar(Car* car);
    void draw(glm::mat4 mEyeProjectionMatrix, glm::mat4 mEyeViewMatrix, glm::vec4 myLighPosInEyeSpace, glm::vec3 mCameraPos);
private:
    Car* car;
    vector<Base*>* bases;
    glm::mat4 mCamera;
    glm::vec3 camPos;
    glm::vec4 updateLight(glm::mat4 viewMatrix, glm::vec3 xyz);
};


#endif //PHYVR_RENDERER_H
