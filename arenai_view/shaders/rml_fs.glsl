#version 330 core

precision mediump float;

// RmlUi hands premultiplied-alpha colors: untextured geometry is drawn with a
// 1x1 white texture bound, so a single program covers both cases
uniform sampler2D u_texture;

in vec4 v_color;
in vec2 v_tex_coord;

out vec4 fragColor;

void main() {
    fragColor = v_color * texture(u_texture, v_tex_coord);
}
