//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <vector>
#include <glm/glm.hpp>
#include <android/asset_manager.h>
#include "../levels/level.h"

class Renderer {
public:
	Renderer(Level *level);

	void update(glm::mat4 mHeadView, bool VR);

	void
	draw(glm::mat4 mEyeProjectionMatrix, glm::mat4 mEyeViewMatrix, glm::vec4 myLighPosInEyeSpace, glm::vec3 mCameraPos);

private:
	Level *level;
	glm::mat4 mCamera;
	glm::vec3 camPos;
	glm::vec3 camLookAtVec; // normalized Vec
	glm::vec3 camUpVec;

	glm::vec4 updateLight(glm::mat4 viewMatrix, glm::vec3 xyz);
};


#endif //PHYVR_RENDERER_H
