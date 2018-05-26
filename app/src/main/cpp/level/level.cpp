//
// Created by samuel on 25/05/18.
//

#include "level.h"

#include <glm/gtc/matrix_transform.hpp>
#include <GLES2/gl2.h>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "../utils/assets.h"

Level::Level(AAssetManager* mgr) {
    this->mgr = mgr;
    std::string objTxt = getFileText(mgr, "obj/icosahedron.obj");
    float* color = new float[4]{1.f, 0.f, 0.f, 1.f};
    this->modelVBO = new ModelVBO(objTxt, color);
}

void Level::init() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);

    glClearColor(0.5f, 0.5f, 0.5f, 1.f);
}

void Level::update(glm::mat4 mHeadView) {

}

void Level::draw(glm::mat4 mEyeProjectionMatrix,
                 glm::mat4 mEyeViewMatrix,
                 glm::vec4 myLighPosInEyeSpace,
                 glm::vec3 mCameraPos) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mCamera = glm::lookAt(glm::vec3(0.f),
                          glm::vec3(0.f, 0.f, 1.f),
                          glm::vec3(0.f, 1.f, 0.f));

    glm::mat4 mViewMatrix = mEyeViewMatrix * mCamera;

    glm::mat4 model(1.f);
    model = glm::translate(model, glm::vec3(0.f, 0.f, 5.f));

    glm::mat4 mvMatrix = mViewMatrix * model;
    glm::mat4 mvpMatrix = mEyeProjectionMatrix * mvMatrix;

    glm::vec4 lighPos = this->updateLight(mViewMatrix, glm::vec3(0.f));

    modelVBO->draw(mvpMatrix, mvMatrix, glm::vec3(lighPos[0], lighPos[1], lighPos[2]));
}

glm::vec4 Level::updateLight(glm::mat4 viewMatrix, glm::vec3 xyz) {
    glm::mat4 lightModel(1.f);
    lightModel = glm::translate(lightModel, xyz);
    glm::vec4 lightPosInModelSpace(0.f, 0.f, 0.f, 1.f);
    return viewMatrix * lightModel * lightPosInModelSpace;

}

Level::~Level() {
    delete modelVBO;
}
