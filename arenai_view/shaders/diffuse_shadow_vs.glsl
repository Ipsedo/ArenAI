#version 330 core

precision highp float;
precision highp int;

uniform mat4 u_mvp_matrix;
uniform mat4 u_mv_matrix;
uniform mat4 u_shadow_mvp_matrix;

in vec3 a_position;

out vec3 v_position;
out vec4 v_shadow_coord;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));
    v_shadow_coord = u_shadow_mvp_matrix * vec4(a_position, 1.0);
    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
