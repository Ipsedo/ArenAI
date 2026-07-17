#version 450

layout(push_constant) uniform Push { mat4 u_mvp_matrix; };

layout(location = 0) in vec3 a_vp;

layout(location = 0) out vec3 v_tex_coords;

void main() {
    v_tex_coords = a_vp;
    gl_Position = u_mvp_matrix * vec4(a_vp, 1.0);
}
