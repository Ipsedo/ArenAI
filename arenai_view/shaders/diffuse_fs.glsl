#version 450

// per-frame globals, bound once per renderer frame (set 0 of every scene pipeline)
layout(set = 0, binding = 0, std140) uniform FrameGlobals {
    vec4 u_light_pos;// light position in view space (w unused)
    vec4 u_world_up; // world up axis in view space (xyz) and camera world height (w)
    vec4 u_fog_color;// rgb only
};

layout(set = 1, binding = 0, std140) uniform Material { vec4 u_color; };

layout(location = 0) in vec3 v_position;

layout(location = 0) out vec4 fragColor;

// cool tint of the unlit facets, instead of plain black shading
const vec3 SHADOW_TINT = vec3(0.42, 0.44, 0.60);
// exponential fog, calibrated for a far plane of ~3500 units
const float FOG_DENSITY = 0.0014;

void main() {
    // flat shading: face normal from screen-space derivatives of the
    // view-space position, so smooth vertex normals are not needed; dFdy
    // follows Vulkan's downward window y (GL: upward), hence the swapped
    // operands to keep the normal facing outward
    vec3 normal = normalize(cross(dFdy(v_position), dFdx(v_position)));

    vec3 light_vector = normalize(u_light_pos.xyz - v_position);

    // wrapped (half-Lambert) diffuse: soft falloff, no pitch-black faces;
    // cubed to keep enough contrast between lit and unlit facets
    float wrap = dot(normal, light_vector) * 0.5 + 0.5;
    float light_amount = wrap * wrap * wrap;

    vec3 base = u_color.rgb;
    vec3 color = mix(base * SHADOW_TINT, base, light_amount);

    // distance fog toward the sky color
    float fog = 1.0 - exp(-FOG_DENSITY * length(v_position));
    color = mix(color, u_fog_color.rgb, fog);

    fragColor = vec4(color, u_color.a);
}
