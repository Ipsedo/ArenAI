//
// Created by samuel on 26/03/2023.
//

#include "./hud.h"
#include "../utils/logging.h"

#include <glm/gtx/transform.hpp>
#include <utility>

HUDDrawable::~HUDDrawable() {}

JoyStickDrawable::JoyStickDrawable(AAssetManager *mgr,
                                   std::function<joystick(void)> get_input_px,
                                   glm::vec2 center_px, float size_px,
                                   float stick_size_px)
    : get_input(std::move(get_input_px)), center_x(center_px.x),
      center_y(center_px.y), size(size_px), stick_size(stick_size_px) {

  program =
      Program::Builder(mgr, "shaders/simple_vs.glsl", "shaders/simple_fs.glsl")
          .add_uniform("u_color")
          .add_uniform("u_mvp_matrix")
          .add_attribute("a_position")
          .add_buffer("square_buffer", {-1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f,
                                        -1.f, 0.f, -1.f, -1.f, 0.f})
          .build();
}

void JoyStickDrawable::draw(int width, int height) {
  float ratio = float(width) / float(height);

  float center_x_rel = ratio * ((center_x / float(width)) * 2.f - 1.f);
  float center_y_rel = (center_y / float(height)) * 2.f - 1.f;

  auto [stick_x, stick_y] = get_input();
  float stick_x_rel = ratio * ((stick_x / float(width)) * 2.f - 1.f);
  float stick_y_rel = (stick_y / float(height)) * 2.f - 1.f;

  float size_rel = size / float(width) * ratio;
  float stick_size_rel = stick_size / float(width) * ratio;

  glm::mat4 v_matrix =
      glm::lookAt(glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 0.f),
                  glm::vec3(0.f, 1.f, 0.f));
  glm::mat4 p_matrix =
      glm::ortho(-1.f * ratio, 1.f * ratio, -1.f, 1.f, -1.f, 1.f);
  glm::mat4 vp_matrix = p_matrix * v_matrix;

  glm::mat4 bounds_m_matrix =
      glm::translate(glm::vec3(center_x_rel, center_y_rel, 0.f)) *
      glm::scale(glm::vec3(size_rel, size_rel, 1.f));
  glm::mat4 stick_m_matrix =
      glm::translate(glm::vec3(stick_x_rel, stick_y_rel, 0.f)) *
      glm::scale(glm::vec3(stick_size_rel, stick_size_rel, 1.f));

  glLineWidth(5.f);

  // draw
  program->use();

  program->attrib("a_position", "square_buffer", 3, 3 * 4, 0);
  program->uniform_mat4("u_mvp_matrix", vp_matrix * bounds_m_matrix);
  program->uniform_vec4("u_color", glm::vec4(1., 0, 0, 1.));
  Program::draw_arrays(GL_LINE_LOOP, 0, 4);

  program->attrib("a_position", "square_buffer", 3, 3 * 4, 0);
  program->uniform_mat4("u_mvp_matrix", vp_matrix * stick_m_matrix);
  program->uniform_vec4("u_color", glm::vec4(1., 0., 0., 1.));
  Program::draw_arrays(GL_LINE_LOOP, 0, 4);

  program->disable_attrib_array();

  glLineWidth(1.f);
}

JoyStickDrawable::~JoyStickDrawable() { program.reset(); }
