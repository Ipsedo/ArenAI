#version 450

layout(push_constant) uniform Push { mat4 u_mvp_matrix; };

layout(location = 0) in vec4 a_position;

void main() {
    gl_Position = u_mvp_matrix * a_position;
}
