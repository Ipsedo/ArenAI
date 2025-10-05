//
// Created by samuel on 16/08/18.
//

#include "limits.h"

#include "../utils/vec.h"

Limits::~Limits() {}

BoxLimits::BoxLimits(glm::vec3 startVert, glm::vec3 edgeSize)
    : startVert(startVert), edgeSize(edgeSize) {}

bool BoxLimits::isInside(Base *b) {
    glm::vec3 pos = btVector3ToVec3(b->getCenterOfMassTransform().getOrigin());
    return pos.x > startVert.x && pos.y > startVert.y && pos.z > startVert.z
           && pos.x < startVert.x + edgeSize.x && pos.y < startVert.y + edgeSize.y
           && pos.z < startVert.z + edgeSize.z;
}
