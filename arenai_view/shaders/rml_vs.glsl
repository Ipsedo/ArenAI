#version 330 core

uniform mat4 u_projection;
uniform vec2 u_translation;

in vec2 a_position;
in vec4 a_color;
in vec2 a_tex_coord;

out vec4 v_color;
out vec2 v_tex_coord;

void main() {
    v_color = a_color;
    v_tex_coord = a_tex_coord;
    gl_Position = u_projection * vec4(a_position + u_translation, 0.0, 1.0);
}
