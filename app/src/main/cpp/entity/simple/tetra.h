//
// Created by samuel on 21/04/19.
//

#ifndef PHYVR_TETRA_H
#define PHYVR_TETRA_H

#include "../poly.h"


class Tetra : public Poly {
public:
	Tetra(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);

	Tetra(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);
};


#endif //PHYVR_TETRA_H
