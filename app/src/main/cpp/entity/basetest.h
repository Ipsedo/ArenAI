//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_BASETEST_H
#define PHYVR_BASETEST_H


#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include "../graphics/drawable/modelvbo.h"

class BaseTest : public btRigidBody {
private:
	glm::vec3 scale;
	ModelVBO modelVBO;

protected:
	BaseTest(const btRigidBodyConstructionInfo &constructionInfo, const ModelVBO &modelVBO, const glm::vec3 &scale);

public:
	virtual void update();

	virtual void decreaseLife(int toSub);

	virtual bool isDead();

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos);

	virtual ~BaseTest();
};


#endif //PHYVR_BASETEST_H
