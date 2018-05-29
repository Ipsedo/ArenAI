//
// Created by samuel on 29/05/18.
//

#include "cone.h"
#include "../../utils/assets.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

Cone::Cone(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix,
           float mass) {
    std::string objTxt = getFileText(mgr, "obj/cone.obj");

    modelVBO = new ModelVBO(
            objTxt,
            new float[4]{ (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          1.f});

    this->scale = scale;

    collisionShape.push_back(new btConeShape(1.f, 2.f));
    collisionShape[0]->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    glm::quat tmp = glm::quat_cast(rotationMatrix);
    myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

    btVector3 intertie(0.f, 0.f, 0.f);
    if (mass)
        collisionShape[0]->calculateLocalInertia(mass, intertie);

    defaultMotionState = new btDefaultMotionState(myTransform);

    btRigidBody::btRigidBodyConstructionInfo constrInfo(mass,
                                                        defaultMotionState,
                                                        collisionShape[0],
                                                        intertie);

    rigidBody.push_back(new btRigidBody(constrInfo));
}

void Cone::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
    std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
    modelVBO->draw(std::get<0>(matrixes), std::get<1>(matrixes), lighPos);
}

Cone::~Cone() {
    delete modelVBO;
}
