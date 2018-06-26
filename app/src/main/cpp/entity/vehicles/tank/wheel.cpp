//
// Created by samuel on 26/06/18.
//

#include "wheel.h"
#include "../../../../../../../libs/glm/glm/vec3.hpp"

Wheel::Wheel(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo, const ModelVBO &modelVBO,
			 const glm::vec3 &scale) : BaseTest(constructionInfo, modelVBO, scale) {

}
