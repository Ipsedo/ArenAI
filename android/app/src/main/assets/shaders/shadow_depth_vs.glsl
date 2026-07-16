#version 300 es

precision highp float;
precision highp int;

uniform mat4 u_light_mvp_matrix;

in vec3 a_position;

void main() {
    gl_Position = u_light_mvp_matrix * vec4(a_position, 1.0);
}
