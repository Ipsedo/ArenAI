//
// Created by samuel on 26/05/18.
//

#include "box.h"
#include "../utils/assets.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

Box::Box(AAssetManager* mgr,
         glm::vec3 pos,
         glm::vec3 sideScale,
         glm::mat4 rotationMatrix,
         float mass) {

    std::string objTxt = getFileText(mgr, "obj/cube.obj");

    modelVBO = new ModelVBO(
            objTxt,
            new float[4]{ (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          1.f});

    scale = sideScale;

    collisionShape = new btBoxShape(btVector3(scale.x, scale.y, scale.z));

    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    glm::quat tmp = glm::quat_cast(rotationMatrix);
    myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

    btVector3 intertie(0.f, 0.f, 0.f);
    if (mass)
        collisionShape->calculateLocalInertia(mass, intertie);

    defaultMotionState = new btDefaultMotionState(myTransform);

    btRigidBody::btRigidBodyConstructionInfo constrInfo(mass,
                                                        defaultMotionState,
                                                        collisionShape,
                                                        intertie);

    rigidBody = new btRigidBody(constrInfo);
}

Box::~Box() {
    delete defaultMotionState;
    delete collisionShape;
    delete rigidBody;
}

void Box::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
    btScalar tmp[16];
    defaultMotionState->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale);

    glm::mat4 mvMatrix = vMatrix * modelMatrix;
    glm::mat4 mvpMatrix = pMatrix * mvMatrix;

    modelVBO->draw(mvpMatrix, mvMatrix, lighPos);
}
