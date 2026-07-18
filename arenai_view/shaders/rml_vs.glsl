#version 450

layout(push_constant) uniform Push {
    mat4 u_projection;
    vec2 u_translation;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_tex_coord;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coord;

void main() {
    v_color = a_color;
    v_tex_coord = a_tex_coord;
    gl_Position = u_projection * vec4(a_position + u_translation, 0.0, 1.0);
}
