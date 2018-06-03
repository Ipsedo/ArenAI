//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CAR_H
#define PHYVR_CAR_H

#include <glm/glm.hpp>
#include <android/asset_manager.h>
#include <btBulletDynamicsCommon.h>
#include "../player.h"

class Car : public Player {
public:
	Car(glm::vec3 pos, btDynamicsWorld *world, AAssetManager *mgr);

	void onInput(input in) override;

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

	glm::vec3 camPos(bool VR) override;
	glm::vec3 camLookAtVec(bool VR) override;
	glm::vec3 camUpVec(bool VR) override;

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
