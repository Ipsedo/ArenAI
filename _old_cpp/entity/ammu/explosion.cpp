//
// Created by samuel on 20/08/18.
//

#include "explosion.h"

#include "btBulletDynamicsCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

#define MAX_FRAMES 5
#define INIT_SIZE 1e-1f
#define MAX_SIZE 60.f
#define BASE_SIZE exp(log(MAX_SIZE) / MAX_FRAMES)
#define MASS 0.f

float getSize(int frame) {
  return 6.f;// INIT_SIZE + float(pow(BASE_SIZE, frame - 1));
}

Explosion::Explosion(btVector3 pos, DiffuseModel *modelVBO)
    : Poly(
      [](glm::vec3 scale) {
        btCollisionShape *shape = new btSphereShape(1.f);
        shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
        return shape;
      },
      modelVBO, glm::vec3(pos.x(), pos.y(), pos.z()), glm::vec3(getSize(0)), glm::mat4(1.f), MASS,
      false),
      nbFrames(MAX_FRAMES) {}

bool Explosion::isDead() { return nbFrames <= 0; }

void Explosion::update() {
  scale = glm::vec3(getSize(MAX_FRAMES - nbFrames));
  getCollisionShape()->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
  nbFrames--;
}

bool Explosion::needExplosion() { return false; }

void Explosion::onContactFinish(Base *other) {
  // Base::onContactFinish(other);
  btVector3 vec = other->getCenterOfMassPosition() - getCenterOfMassPosition();
  vec = vec.normalized();

  float force = 2.184e1f / other->getInvMass();

  other->activate(true);
  other->applyCentralImpulse(force * vec);
}

Particules::Particules(btVector3 explosionCenter, DiffuseModel *triangle)
    : Sphere(
      triangle, glm::vec3(explosionCenter.x(), explosionCenter.y(), explosionCenter.z()),
      glm::vec3(0.25f),
      glm::rotate(
        glm::mat4(1.f), 2.f * float(M_PI) * float(rand()) / float(RAND_MAX),
        glm::normalize(glm::vec3(
          float(rand()) / float(RAND_MAX), float(rand()) / float(RAND_MAX),
          float(rand()) / float(RAND_MAX)))),
      300.f),
      nbFrames(15) {

  this->activate(true);

  float phi = float(M_PI) * float(rand()) / float(RAND_MAX);
  float theta = 2.f * float(M_PI) * float(rand()) / float(RAND_MAX);

  float x = sin(phi) * cos(theta), y = sin(phi) * sin(theta), z = cos(phi);
  btVector3 dir = btVector3(x, y, z);

  float max_force = 2e6f, min_force = 5e5f;
  float factor =
    (max_force - min_force) * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) + min_force;

  // this->setLinearVelocity(dir * vel);
  this->applyCentralForce(dir * factor);
}

bool Particules::isDead() { return nbFrames <= 0; }

void Particules::update() { nbFrames--; }

bool Particules::needExplosion() { return false; }
