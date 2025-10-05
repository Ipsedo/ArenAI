//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_MISC_H
#define PHYVR_MISC_H

#include "glm/glm.hpp"

struct draw_infos {
  glm::mat4 proj_matrix;
  glm::mat4 view_matrix;
  glm::vec3 light_pos;
  glm::vec3 camera_pos;
  bool VR;
};

class Drawable {
public:
  virtual void draw(draw_infos infos) = 0;
};

struct gl_draw_info {
  glm::mat4 mvp_matrix;
  glm::mat4 mv_matrix;
  glm::vec3 light_pos;
  glm::vec3 camera_pos;
};

class GLDrawable {
public:
  virtual void draw(gl_draw_info infos) = 0;

  virtual ~GLDrawable() {}
};

#endif// PHYVR_MISC_H
