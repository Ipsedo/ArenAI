//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_CONE_H
#define PHYVR_CONE_H


#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../base.h"

class Cone : public Base {
public:
    Cone(AAssetManager* mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;
    ~Cone();
private:
    ModelVBO* modelVBO;
};


#endif //PHYVR_CONE_H
