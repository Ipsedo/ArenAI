//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_MISSILE_H
#define PHYVR_MISSILE_H


#include "../simple/cone.h"

class Missile : public Cone {
public:
	Missile(DiffuseModel *modelVBO, const glm::vec3 &pos, const glm::vec3 &scale,
			const glm::mat4 &rotMat, float mass,
			int life);

	void decreaseLife(int toSub) override;

	bool isDead() override;

private:
	int life;
};

#endif //PHYVR_MISSILE_H
