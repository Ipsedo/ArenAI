//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_DRAWABLE_H
#define PHYVR_DRAWABLE_H

#include <glm/glm.hpp>

class Drawable {
public:
    virtual void draw(
            glm::mat4 mvp_matrix,
            glm::mat4 mv_matrix,
            glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos
    ) = 0;
};

#endif //PHYVR_DRAWABLE_H
