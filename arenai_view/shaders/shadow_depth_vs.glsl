#version 450

layout(push_constant) uniform Push { mat4 u_light_mvp_matrix; };

layout(location = 0) in vec3 a_position;

void main() {
    gl_Position = u_light_mvp_matrix * vec4(a_position, 1.0);
}
