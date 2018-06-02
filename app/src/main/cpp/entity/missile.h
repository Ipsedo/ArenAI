//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_MISSILE_H
#define PHYVR_MISSILE_H


#include "poly/cone.h"

class Missile : public Cone {
public:
	Missile(ModelVBO* modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);
	void update() override;
	bool isDead() override;
};


#endif //PHYVR_MISSILE_H
