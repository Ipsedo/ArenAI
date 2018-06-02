//
// Created by samuel on 02/06/18.
//

#include "missile.h"

Missile::Missile(ModelVBO *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass) : Cone(
		modelVBO, pos, scale, rotationMatrix, mass) {
}

void Missile::update() {

}

bool Missile::isDead() {
	return rigidBody[0]->getLinearVelocity().norm() < 1e-2f;
}
