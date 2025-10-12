//
// Created by samuel on 26/03/2023.
//

#include <phyvr_view/hud.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <utility>

#include <glm/gtx/transform.hpp>

HUDDrawable::~HUDDrawable() = default;

std::vector<float> get_circle_points_(const int nb_points) {
    std::vector<float> circle_points{};
    for (int i = 0; i < nb_points; i++) {
        const double angle = static_cast<double>(i) * M_PI * 2. / static_cast<double>(nb_points);

        circle_points.push_back(static_cast<float>(cos(angle)));
        circle_points.push_back(static_cast<float>(sin(angle)));
        circle_points.push_back(0.f);
    }

    return circle_points;
}

/*
 * ButtonDrawable
 */

ButtonDrawable::ButtonDrawable(
    const std::shared_ptr<AbstractFileReader> &file_reader, std::function<button(void)> get_input,
    const glm::vec2 center_px, const float size_px)
    : get_input(std::move(get_input)), center_x(center_px.x), center_y(center_px.y), size(size_px),
      nb_points(128) {

    program = Program::Builder(file_reader, "shaders/simple_vs.glsl", "shaders/simple_fs.glsl")
                  .add_uniform("u_color")
                  .add_uniform("u_mvp_matrix")
                  .add_attribute("a_position")
                  .add_buffer("circle_buffer", get_circle_points_(nb_points))
                  .build();
}

void ButtonDrawable::draw(const int width, const int height) {
    const float ratio = static_cast<float>(width) / static_cast<float>(height);

    const float center_x_rel = ratio * (center_x / static_cast<float>(width) * 2.f - 1.f);
    const float center_y_rel = center_y / static_cast<float>(height) * 2.f - 1.f;

    auto [pressed] = get_input();

    const float size_rel = size / static_cast<float>(width) * ratio;

    const glm::mat4 v_matrix =
        glm::lookAt(glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 p_matrix = glm::ortho(-1.f * ratio, 1.f * ratio, -1.f, 1.f, -1.f, 1.f);
    const glm::mat4 vp_matrix = p_matrix * v_matrix;

    const glm::mat4 button_m_matrix = glm::translate(glm::vec3(center_x_rel, center_y_rel, 0.f))
                                      * glm::scale(glm::vec3(size_rel, size_rel, 1.f));

    glLineWidth(pressed ? 8.f : 5.f);

    // on_draw
    program->use();

    program->attrib("a_position", "circle_buffer", 3, 3 * 4, 0);
    program->uniform_mat4("u_mvp_matrix", vp_matrix * button_m_matrix);
    program->uniform_vec4("u_color", glm::vec4(1., 0., 0., 1.));
    Program::draw_arrays(GL_LINE_LOOP, 0, nb_points);

    program->disable_attrib_array();

    glLineWidth(1.f);
}

/*
 * JoyStickDrawable
 */

JoyStickDrawable::JoyStickDrawable(
    const std::shared_ptr<AbstractFileReader> &file_reader,
    std::function<joystick(void)> get_input_px, const glm::vec2 center_px, const float size_px,
    const float stick_size_px)
    : get_input(std::move(get_input_px)), center_x(center_px.x), center_y(center_px.y),
      size(size_px), stick_size(stick_size_px), nb_point_bound(4), nb_point_stick(128) {

    program =
        Program::Builder(file_reader, "shaders/simple_vs.glsl", "shaders/simple_fs.glsl")
            .add_uniform("u_color")
            .add_uniform("u_mvp_matrix")
            .add_attribute("a_position")
            .add_buffer(
                "square_buffer", {-1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, -1.f, 0.f, -1.f, -1.f, 0.f})
            .add_buffer("circle_buffer", get_circle_points_(nb_point_stick))
            .build();
}

void JoyStickDrawable::draw(const int width, const int height) {
    const float ratio = static_cast<float>(width) / static_cast<float>(height);

    const float center_x_rel = ratio * (center_x / static_cast<float>(width) * 2.f - 1.f);
    const float center_y_rel = center_y / static_cast<float>(height) * 2.f - 1.f;

    auto [stick_x, stick_y] = get_input();
    const float stick_x_rel = ratio * (stick_x / static_cast<float>(width) * 2.f - 1.f);
    const float stick_y_rel = stick_y / static_cast<float>(height) * 2.f - 1.f;

    const float size_rel = size / static_cast<float>(width) * ratio;
    const float stick_size_rel = stick_size / static_cast<float>(width) * ratio;

    const glm::mat4 v_matrix =
        glm::lookAt(glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 p_matrix = glm::ortho(-1.f * ratio, 1.f * ratio, -1.f, 1.f, -1.f, 1.f);
    const glm::mat4 vp_matrix = p_matrix * v_matrix;

    const glm::mat4 bounds_m_matrix = glm::translate(glm::vec3(center_x_rel, center_y_rel, 0.f))
                                      * glm::scale(glm::vec3(size_rel, size_rel, 1.f));
    const glm::mat4 stick_m_matrix = glm::translate(glm::vec3(stick_x_rel, stick_y_rel, 0.f))
                                     * glm::scale(glm::vec3(stick_size_rel, stick_size_rel, 1.f));

    glLineWidth(5.f);

    // on_draw
    program->use();

    program->attrib("a_position", "square_buffer", 3, 3 * 4, 0);
    program->uniform_mat4("u_mvp_matrix", vp_matrix * bounds_m_matrix);
    program->uniform_vec4("u_color", glm::vec4(1., 0, 0, 1.));
    Program::draw_arrays(GL_LINE_LOOP, 0, nb_point_bound);

    program->attrib("a_position", "circle_buffer", 3, 3 * 4, 0);
    program->uniform_mat4("u_mvp_matrix", vp_matrix * stick_m_matrix);
    program->uniform_vec4("u_color", glm::vec4(1., 0., 0., 1.));
    Program::draw_arrays(GL_LINE_LOOP, 0, nb_point_stick);

    program->disable_attrib_array();

    glLineWidth(1.f);
}
