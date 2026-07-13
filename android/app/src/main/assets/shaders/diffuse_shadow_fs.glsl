#version 300 es

precision highp float;
precision highp int;

uniform vec3 u_light_pos;
uniform vec4 u_color;
uniform vec3 u_fog_color;

uniform highp sampler2DShadow u_shadow_map;

in vec3 v_position;
in vec4 v_shadow_coord;

out vec4 fragColor;

const vec2 POISSON[12] = vec2[](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457),
    vec2(-0.203, 0.621), vec2(0.962, -0.195), vec2(0.473, -0.480),
    vec2(0.519, 0.767), vec2(0.185, -0.893), vec2(0.507, 0.064),
    vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

// fraction of the light kept in fully shadowed areas, so that the object's
// own color still shows through instead of going black
const float SHADOW_MIN_LIGHT = 0.35;

// cool tint of the unlit facets, instead of plain black shading
const vec3 SHADOW_TINT = vec3(0.42, 0.44, 0.60);
// exponential fog, calibrated for a far plane of ~3500 units
const float FOG_DENSITY = 0.0014;

// fraction of the shadow-map border over which shadows fade out, so that the
// edge of the shadow frustum stays invisible instead of cutting sharply
const float EDGE_FADE = 0.15;

float shadow_factor() {
    vec3 coord = v_shadow_coord.xyz / v_shadow_coord.w;

    // behind the far plane of the shadow frustum: fully lit
    if (coord.z > 1.0) return 1.0;

    // fade toward "fully lit" near the border of the shadow map
    vec2 to_edge = min(coord.xy, 1.0 - coord.xy);
    float fade = smoothstep(0.0, EDGE_FADE, min(to_edge.x, to_edge.y));
    if (fade <= 0.0) return 1.0;

    // constant bias; slope-scaled acne removal is handled by glPolygonOffset
    // during the depth pass
    const float bias = 0.0005;
    const float radius = 3.5;
    vec2 texel_size = 1.0 / vec2(textureSize(u_shadow_map, 0));

    // random per-pixel rotation of the Poisson disk turns banding into noise
    float angle =
        6.2831853 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float c = cos(angle), s = sin(angle);
    mat2 rotation = mat2(c, s, -s, c);

    float lit = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 offset = rotation * POISSON[i] * radius * texel_size;
        lit += texture(u_shadow_map, vec3(coord.xy + offset, coord.z - bias));
    }
    return mix(1.0, lit / 12.0, fade);
}

void main() {
    // flat shading: face normal from screen-space derivatives of the
    // view-space position, so smooth vertex normals are not needed
    vec3 normal = normalize(cross(dFdx(v_position), dFdy(v_position)));

    vec3 light_vector = normalize(u_light_pos - v_position);

    // wrapped (half-Lambert) diffuse: soft falloff, no pitch-black faces;
    // cubed to keep enough contrast between lit and unlit facets
    float wrap = dot(normal, light_vector) * 0.5 + 0.5;
    float light_amount = wrap * wrap * wrap;

    // cast shadows attenuate the light instead of removing it
    light_amount *= mix(SHADOW_MIN_LIGHT, 1.0, shadow_factor());

    vec3 base = u_color.rgb;
    vec3 color = mix(base * SHADOW_TINT, base, light_amount);

    // distance fog toward the sky color
    float fog = 1.0 - exp(-FOG_DENSITY * length(v_position));
    color = mix(color, u_fog_color, fog);

    fragColor = vec4(color, u_color.a);
}
