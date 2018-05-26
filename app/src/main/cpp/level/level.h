//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H

#include <glm/glm.hpp>
#include <android/asset_manager.h>

#include "../drawable/modelvbo.h"

class Level {
public:
    Level(AAssetManager* mgr);
    void init();
    void update(glm::mat4 mHeadView);
    void draw(glm::mat4 mEyeProjectionMatrix, glm::mat4 mEyeViewMatrix, glm::vec4 myLighPosInEyeSpace, glm::vec3 mCameraPos);
    ~Level();
private:
    AAssetManager* mgr;
    ModelVBO* modelVBO;
    glm::mat4 mCamera;
    glm::vec4 updateLight(glm::mat4 viewMatrix, glm::vec3 xyz);
};


#endif //PHYVR_LEVEL_H
