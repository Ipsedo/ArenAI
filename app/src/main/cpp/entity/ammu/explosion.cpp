//
// Created by samuel on 20/08/18.
//

#include "explosion.h"
#include <math.h>
#include <glm/glm.hpp>
#include <btBulletDynamicsCommon.h>

#define MAX_FRAMES 5
#define INIT_SIZE 1e-1f
#define MAX_SIZE 60.f
#define BASE_SIZE exp(log(MAX_SIZE) / MAX_FRAMES)
#define MASS 0.f

float getSize(int frame) {
	return 6.f;//INIT_SIZE + float(pow(BASE_SIZE, frame - 1));
}

auto l = [](glm::vec3 scale) {
	btCollisionShape *shape = new btSphereShape(1.f);
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	return shape;
};

Explosion::Explosion(btVector3 pos, DiffuseModel *modelVBO)
		: Poly(l, modelVBO, glm::vec3(pos.x(), pos.y(), pos.z()), glm::vec3(getSize(0)),
			   glm::mat4(1.f), MASS, false), nbFrames(MAX_FRAMES) {}

bool Explosion::isDead() {
	return nbFrames <= 0;
}

void Explosion::update() {
	scale = glm::vec3(getSize(MAX_FRAMES - nbFrames));
	getCollisionShape()->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	nbFrames--;
}

bool Explosion::needExplosion() {
	return false;
}

void Explosion::onContactFinish(Base *other) {
	//Base::onContactFinish(other);
	btVector3 vec = other->getCenterOfMassPosition() - getCenterOfMassPosition();
	vec = vec.normalized();

	float force = 2.184e1f / other->getInvMass();

	other->activate(true);
	other->applyCentralImpulse(force * vec);
}
