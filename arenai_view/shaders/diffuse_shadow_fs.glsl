#version 330 core

precision highp float;
precision highp int;

uniform vec3 u_light_pos;
uniform vec4 u_color;
uniform vec3 u_fog_color;
// world up axis expressed in view space (xyz) and camera world height (w):
// world_height(p_view) = dot(xyz, p_view) + w
uniform vec4 u_world_up;

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

// hemisphere ambient: unlit facets take a cool sky tint when facing up and a
// warmer ground-bounce tint when facing down, giving cheap volume to the
// flat-shaded geometry
const vec3 SHADOW_TINT_SKY = vec3(0.40, 0.45, 0.66);
const vec3 SHADOW_TINT_GROUND = vec3(0.47, 0.43, 0.41);

// height fog: base density at height 0, decaying exponentially with world
// height so the fog settles in the valleys; calibrated for a far plane of
// ~3500 units (the SSAO pass reuses FOG_DENSITY to fade with it)
const float FOG_DENSITY = 0.0014;
const float FOG_HEIGHT_FALLOFF = 0.0035;

// Blinn-Phong sun glint: each flat facet catches the light as one block
const vec3 SPECULAR_COLOR = vec3(1.0, 0.97, 0.90);
const float SPECULAR_STRENGTH = 0.22;
const float SPECULAR_POWER = 32.0;

// fresnel rim light in the sky color, detaching silhouettes from the ground
const float RIM_STRENGTH = 0.18;
const float RIM_POWER = 3.0;

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

// optical depth of an exponential height-fog profile, integrated exactly
// along the camera→fragment ray (the camera sits at the view-space origin)
float fog_amount() {
    float dist = length(v_position);
    float camera_height = u_world_up.w;
    float fragment_height = dot(u_world_up.xyz, v_position) + u_world_up.w;

    float t = FOG_HEIGHT_FALLOFF * (fragment_height - camera_height);
    float height_integral = abs(t) > 1e-3 ? (1.0 - exp(-t)) / t : 1.0 - 0.5 * t;

    float optical_depth =
        FOG_DENSITY * exp(-FOG_HEIGHT_FALLOFF * camera_height) * height_integral * dist;
    return 1.0 - exp(-optical_depth);
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
    float shadow = shadow_factor();
    light_amount *= mix(SHADOW_MIN_LIGHT, 1.0, shadow);

    // hemisphere ambient tint, blended on how much the facet faces the sky
    float up_amount = dot(normal, u_world_up.xyz) * 0.5 + 0.5;
    vec3 shadow_tint = mix(SHADOW_TINT_GROUND, SHADOW_TINT_SKY, up_amount);

    // the facet shading is calibrated directly in display (sRGB) space:
    // mixing in linear space would crush the contrast between lit and unlit
    // faces; the post pass handles the linear-space tonemapping and grading
    vec3 base = u_color.rgb;
    vec3 color = mix(base * shadow_tint, base, light_amount);

    vec3 view_vector = normalize(-v_position);

    // per-facet sun glint, gone in cast shadow and on faces away from the sun
    vec3 half_vector = normalize(light_vector + view_vector);
    float facing = max(dot(normal, light_vector), 0.0);
    float spec = pow(max(dot(normal, half_vector), 0.0), SPECULAR_POWER);
    color += SPECULAR_STRENGTH * spec * facing * shadow * SPECULAR_COLOR;

    // sky-tinted fresnel rim (camera sits at the view-space origin)
    float rim = pow(1.0 - max(dot(normal, view_vector), 0.0), RIM_POWER);
    color += RIM_STRENGTH * rim * u_fog_color;

    // height fog toward the sky color
    color = mix(color, u_fog_color, fog_amount());

    fragColor = vec4(color, u_color.a);
}
