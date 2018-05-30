//
// Created by samuel on 30/05/18.
//

#ifndef PHYVR_CAMERA_H
#define PHYVR_CAMERA_H

#include <glm/glm.hpp>

class Camera {
public:
	virtual glm::vec3 camPos() = 0;

	virtual glm::vec3 camLookAtVec() = 0;

	virtual glm::vec3 camUpVec() = 0;
};

#endif //PHYVR_CAMERA_H
