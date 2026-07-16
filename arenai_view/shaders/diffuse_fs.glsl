#version 330 core

precision highp float;
precision highp int;

uniform vec3 u_light_pos;
uniform vec4 u_color;
uniform vec3 u_fog_color;

in vec3 v_position;

out vec4 fragColor;

// cool tint of the unlit facets, instead of plain black shading
const vec3 SHADOW_TINT = vec3(0.42, 0.44, 0.60);
// exponential fog, calibrated for a far plane of ~3500 units
const float FOG_DENSITY = 0.0014;

void main() {
    // flat shading: face normal from screen-space derivatives of the
    // view-space position, so smooth vertex normals are not needed
    vec3 normal = normalize(cross(dFdx(v_position), dFdy(v_position)));

    vec3 light_vector = normalize(u_light_pos - v_position);

    // wrapped (half-Lambert) diffuse: soft falloff, no pitch-black faces;
    // cubed to keep enough contrast between lit and unlit facets
    float wrap = dot(normal, light_vector) * 0.5 + 0.5;
    float light_amount = wrap * wrap * wrap;

    vec3 base = u_color.rgb;
    vec3 color = mix(base * SHADOW_TINT, base, light_amount);

    // distance fog toward the sky color
    float fog = 1.0 - exp(-FOG_DENSITY * length(v_position));
    color = mix(color, u_fog_color, fog);

    fragColor = vec4(color, u_color.a);
}
