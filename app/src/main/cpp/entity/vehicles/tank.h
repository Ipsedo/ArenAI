//
// Created by samuel on 30/05/18.
//

#ifndef PHYVR_TANK_H
#define PHYVR_TANK_H

#include <glm/glm.hpp>
#include <android/asset_manager.h>
#include "../../graphics/camera.h"
#include "../base.h"
#include "../../controls/controls.h"

class Tank : public Base, public Camera, public Controls {
public:
	Tank(btDynamicsWorld *world, AAssetManager *mgr);

	void onInput(input in) override;

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

	glm::vec3 camPos() override;
	glm::vec3 camLookAtVec() override;
	glm::vec3 camUpVec() override;

	~Tank();

private:
	float direction;
	float speed;

	float turretDir;

	void init(btDynamicsWorld *world, AAssetManager *mgr);

	std::vector<btHinge2Constraint *> pHinge2;
	std::vector<ModelVBO *> modelVBOs;
};


#endif //PHYVR_TANK_H
