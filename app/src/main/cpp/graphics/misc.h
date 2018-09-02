//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_MISC_H
#define PHYVR_MISC_H

#include <glm/glm.hpp>

struct draw_infos {
	glm::mat4 proj_matrix;
	glm::mat4 view_matrix;
	glm::vec3 light_pos;
	glm::vec3 camera_pos;
};

class Drawable {
public:
	virtual void draw(draw_infos infos) = 0;
};

#endif //PHYVR_MISC_H
