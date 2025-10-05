//
// Created by samuel on 06/09/18.
//

#include "compass.h"

#include <cmath>

#include "../utils/vec.h"
#include "glm/gtc/matrix_transform.hpp"

Compass::Compass(Base *target) : target(target), triangle(Triangle(1.f, 1.f, 1.f)) {}

void Compass::draw(draw_infos infos) {
    /*float ratio = infos.proj_matrix[1][1] / infos.proj_matrix[0][0];
  glm::mat4 projMat = glm::ortho(-ratio, ratio, -1.f, 1.f, -1.f, 1.f);*/
    glm::mat4 projMat = infos.proj_matrix;
    glm::mat4 viewMat = infos.view_matrix;

    glm::vec3 frontVec = glm::vec3(0.f, 0.f, -1.f) * glm::mat3(viewMat);
    glm::vec3 upVec = glm::vec3(0.f, 1.f, 0.f) * glm::mat3(viewMat);
    glm::vec3 from = -glm::vec3(viewMat[3]) * glm::mat3(viewMat);

    btTransform tr;
    target->getMotionState()->getWorldTransform(tr);
    glm::vec3 fromTo = glm::normalize(btVector3ToVec3(tr.getOrigin()) - from);

    float angleWithFrontVec = acos(glm::dot(frontVec, fromTo));
    if (angleWithFrontVec < M_PI / 6.) return;

    glm::vec3 vecProj = glm::cross(frontVec, glm::cross(fromTo, frontVec));

    float angle = -acos(glm::dot(upVec, vecProj));

    glm::vec3 tmp = vecProj * glm::mat3(glm::inverse(viewMat));

    if (tmp.x > 0.f) angle = -angle;

    glm::mat4 modelMat = glm::rotate(glm::mat4(1.f), angle, glm::vec3(0.f, 0.f, 1.f))
                         * glm::translate(glm::mat4(1.f), glm::vec3(0.f, 7e-2f, 0.2f)) *
                         // TODO position en fonction de frustum
                         glm::scale(glm::mat4(1.f), glm::vec3(5e-3f));

    glm::mat4 mvpMatrix =
        projMat * glm::lookAt(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 1.f, 0.f))
        * modelMat;

    triangle.draw(mvpMatrix);
}
