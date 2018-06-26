//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include "../../basetest.h"
#include <glm/glm.hpp>

static float wheelRadius = 0.8f;
static float wheelWidth = 0.4f;

static float wheelOffset = 0.6f;

static float wheelbaseOffset = 0.1f;

static float wheelMass = 300.f;

class Wheel : public BaseTest {
private:
	Wheel(const btRigidBodyConstructionInfo &constructionInfo, const ModelVBO &modelVBO, const glm::vec3 &scale);

public:
	static Wheel makeWheel();
};


#endif //PHYVR_WHEEL_H
