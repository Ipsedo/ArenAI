#version 450

// RmlUi hands premultiplied-alpha colors: untextured geometry is drawn with a
// 1x1 white texture bound, so a single pipeline covers both cases
layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coord;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = v_color * texture(u_texture, v_tex_coord);
}
