//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_BOX_H
#define PHYVR_BOX_H

#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../graphics/drawable/modelvbo.h"

class Box {
public:
    Box(AAssetManager* mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);
    btRigidBody* rigidBody;
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos);
    ~Box();
private:
    btCollisionShape* collisionShape;
    btDefaultMotionState* defaultMotionState;
    btTransform myTransform;

    ModelVBO* modelVBO;
    glm::vec3 scale;
    glm::mat4 modelMatrix;
};


#endif //PHYVR_CUBE_H
