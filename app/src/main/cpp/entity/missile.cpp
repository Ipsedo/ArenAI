//
// Created by samuel on 02/06/18.
//

#include "missile.h"

Missile::Missile(DiffuseModel *modelVBO, const glm::vec3 &pos, const glm::vec3 &scale, const glm::mat4 &rotMat,
				 float mass, int life) : Cone(modelVBO, pos, scale, rotMat, mass), life(life) {
	setCollisionFlags(getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
}

void Missile::decreaseLife(int toSub) {
	life = life - toSub >= 0 ? life - toSub : 0;
}

bool Missile::isDead() {
	return life <= 0 || getLinearVelocity().norm() < 1.f;
}
