//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H

#include <glm/glm.hpp>

class level {
public:
    void update(glm::mat4 mHeadView);
    void draw(glm::mat4 mEyeProjectionMatrix, glm::mat4 mEyeViewMatrix, glm::vec4 myLighPosInEyeSpace, glm::vec3 mCameraPos);
private:
    void updateLight(glm::vec3 xyz);
};


#endif //PHYVR_LEVEL_H
