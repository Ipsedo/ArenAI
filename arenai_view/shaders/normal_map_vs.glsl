#version 450

layout(push_constant) uniform Push {
    mat4 u_mvp_matrix;
    mat4 u_mv_matrix;
};

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_tex_coord;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec2 v_tex_coord;
layout(location = 2) out vec3 v_normal;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));

    v_tex_coord = a_tex_coord;

    v_normal = normalize(vec3(u_mv_matrix * vec4(a_normal, 0.0)));

    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
