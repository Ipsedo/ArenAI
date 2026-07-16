#version 330 core

precision mediump float;

in vec3 v_tex_coords;
uniform samplerCube u_cube_map;

out vec4 fragColor;

void main() {
    fragColor = texture(u_cube_map, v_tex_coords);
}
