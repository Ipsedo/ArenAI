//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <vector>
#include <glm/glm.hpp>
#include <android/asset_manager.h>

#include "drawable/modelvbo.h"
#include "../entity/box.h"

class Renderer {
public:
    Renderer(vector<Box*>* boxes);
    void update(glm::mat4 mHeadView);
    void draw(glm::mat4 mEyeProjectionMatrix, glm::mat4 mEyeViewMatrix, glm::vec4 myLighPosInEyeSpace, glm::vec3 mCameraPos);
private:
    vector<Box*>* boxes;
    glm::mat4 mCamera;
    glm::vec4 updateLight(glm::mat4 viewMatrix, glm::vec3 xyz);
};


#endif //PHYVR_RENDERER_H
