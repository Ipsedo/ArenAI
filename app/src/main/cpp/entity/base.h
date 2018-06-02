//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_BASE_H
#define PHYVR_BASE_H

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include "../graphics/drawable/modelvbo.h"

class Base {
public:
	std::vector<btRigidBody *> rigidBody;

	virtual void init();

	virtual void update();

	virtual bool isDead();

	virtual std::tuple<glm::mat4, glm::mat4> getMatrixes(glm::mat4 pMatrix, glm::mat4 vMatrix);

	virtual void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) = 0;

	virtual ~Base();

protected:
	std::vector<btCollisionShape *> collisionShape;
	std::vector<btDefaultMotionState *> defaultMotionState;

	std::vector<glm::vec3> scale;
};


#endif //PHYVR_BASE_H
