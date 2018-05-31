//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CAR_H
#define PHYVR_CAR_H


#include "../base.h"
#include <glm/glm.hpp>
#include <android/asset_manager.h>
#include "../../controls/controls.h"
#include "../../graphics/camera.h"

class Car : public Base, public Camera, public Controls {
public:
	Car(glm::vec3 pos, btDynamicsWorld *world, AAssetManager *mgr);

	void onInput(input in) override;

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

	glm::vec3 camPos() override;
	glm::vec3 camLookAtVec() override;
	glm::vec3 camUpVec() override;

	~Car();

private:
	float direction;
	float speed;

	glm::vec3 spawnPos;

	void init(btDynamicsWorld *world, AAssetManager *mgr);

	std::vector<btHinge2Constraint *> pHinge2;
	std::vector<ModelVBO *> modelVBOs;
};


#endif //PHYVR_CAR_H
