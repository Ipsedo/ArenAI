#version 450

layout(push_constant) uniform Push {
    mat4 u_mvp_matrix;
    mat4 u_mv_matrix;
};

// per-draw slice of the renderer's dynamic UBO: a third matrix does not fit
// in the 128-byte push-constant budget guaranteed by the spec
layout(set = 0, binding = 1, std140) uniform ShadowTransform { mat4 u_shadow_mvp_matrix; };

layout(location = 0) in vec3 a_position;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec4 v_shadow_coord;

void main() {
    v_position = vec3(u_mv_matrix * vec4(a_position, 1.0));
    v_shadow_coord = u_shadow_mvp_matrix * vec4(a_position, 1.0);
    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);
}
