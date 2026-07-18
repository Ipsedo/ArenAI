#version 450

layout(push_constant) uniform Push {
    mat4 u_mvp_matrix;
    mat4 u_mv_matrix;
};

layout(location = 0) in vec3 a_position;

layout(location = 0) out vec3 v_position;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));
    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
