#version 300 es

precision highp float;
precision highp int;

uniform mat4 u_mvp_matrix;
uniform mat4 u_mv_matrix;

in vec3 a_position;

out vec3 v_position;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));
    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
