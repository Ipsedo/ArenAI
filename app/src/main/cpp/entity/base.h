//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_BASETEST_H
#define PHYVR_BASETEST_H


#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include "../graphics/drawable/modelvbo.h"

class Base : public btRigidBody {
private:
	glm::vec3 scale;
	DiffuseModel *modelVBO;

protected:
	Base(const btRigidBodyConstructionInfo &constructionInfo,
			 DiffuseModel *modelVBO, const glm::vec3 &scale);

public:
	virtual void update();

	virtual void decreaseLife(int toSub);

	virtual bool isDead();

	virtual void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos);

	virtual ~Base();
};


#endif //PHYVR_BASETEST_H
