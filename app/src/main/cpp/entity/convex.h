//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_CONVEX_H
#define PHYVR_CONVEX_H


#include "base.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>

class Convex : public Base {
public:
    Convex(AAssetManager* mgr, std::string objFileName,
           glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lightPos) override;
    ~Convex();
private:
    ModelVBO* modelVBO;
    btConvexHullShape* parseObj(std::string objFileText);
};


#endif //PHYVR_CONVEX_H
