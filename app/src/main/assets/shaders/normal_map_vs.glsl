#version 330

uniform mat4 u_mvp_matrix;
uniform mat4 u_mv_matrix;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_tex_coord;

varying vec3 v_position;
varying vec2 v_tex_coord;
varying vec3 v_normal;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));

    v_tex_coord = a_tex_coord;

    v_normal = normalize(vec3(u_mv_matrix * vec4(a_normal, 0.0)));

    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
