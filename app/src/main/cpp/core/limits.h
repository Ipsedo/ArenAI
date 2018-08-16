//
// Created by samuel on 16/08/18.
//

#ifndef PHYVR_LIMITS_H
#define PHYVR_LIMITS_H


#include "../entity/base.h"
#include <glm/glm.hpp>

class Limits {
public:
	virtual bool isInside(Base *b) = 0;
	virtual ~Limits();
};

class BoxLimits : public Limits {
private:
	glm::vec3 startVert;
	glm::vec3 edgeSize;

public:
	BoxLimits(glm::vec3 startVert, glm::vec3 edgeSize);
	bool isInside(Base *b) override ;
};


#endif //PHYVR_LIMITS_H
