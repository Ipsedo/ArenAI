//
// Created by samuel on 13/08/18.
//

#ifndef PHYVR_CHASSIS_H
#define PHYVR_CHASSIS_H


#include "entity/base.h"
#include "../../../controls/controls.h"
#include "../../../graphics/camera.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>

static const glm::vec3 chassisScale = glm::vec3(1.2f, 0.5f, 2.f);
static const float chassisMass = 50000.f;
static float chassisColor[4]{150.f / 255.f, 40.f / 255.f, 27.f / 255.f, 1.f};

class Chassis : public Base, public Controls, public Camera {
private:
	bool respawn;
	btVector3 pos;

public:
	Chassis(const btRigidBodyConstructionInfo &constructionInfo, btDefaultMotionState *motionState,
			ModelVBO *modelVBO, const glm::vec3 &scale, const btVector3 pos);

	void onInput(input in) override;

	void update() override;

	glm::vec3 camPos(bool VR) override;

	glm::vec3 camLookAtVec(bool VR) override;

	glm::vec3 camUpVec(bool VR) override;
};

Chassis *makeChassis(AAssetManager *mgr, btVector3 pos);

#endif //PHYVR_CHASSIS_H
