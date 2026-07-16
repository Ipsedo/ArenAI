#version 330 core

in vec3 a_vp;
uniform mat4 u_mvp_matrix;
out vec3 v_tex_coords;

void main() {
    v_tex_coords = a_vp;
    gl_Position = u_mvp_matrix * vec4(a_vp, 1.0);
}
