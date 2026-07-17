#version 450

layout(set = 1, binding = 0) uniform samplerCube u_cube_map;

layout(location = 0) in vec3 v_tex_coords;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(u_cube_map, v_tex_coords);
}
