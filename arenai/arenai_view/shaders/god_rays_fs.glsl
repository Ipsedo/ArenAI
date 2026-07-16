#version 330 core

precision highp float;

uniform highp sampler2D u_depth;
// projection terms: xy = (proj[0][0], proj[1][1]), zw = (proj[2][2], proj[3][2])
uniform vec4 u_proj_info;
// sun position in uv space, and viewport aspect ratio (width / height)
uniform vec2 u_sun_uv;
uniform float u_aspect;

in vec2 v_uv;

out vec4 fragColor;

const int NB_STEPS = 24;
// fraction of the pixel→sun segment covered by the march
const float MARCH_LENGTH = 0.9;
const float DECAY = 0.92;
// tightness of the sky glow around the sun that feeds the rays
const float GLOW_SHARPNESS = 7.0;
// geometry further than this counts as sky (the skybox writes real depth,
// so a plain far-plane test would miss it)
const float SKY_DISTANCE = 900.0;

float view_z(vec2 uv) {
    float ndc_z = texture(u_depth, uv).r * 2.0 - 1.0;
    return -u_proj_info.w / (ndc_z + u_proj_info.z);
}

// light emitted at this pixel: visible sky, weighted by proximity to the sun
float sky_glow(vec2 uv) {
    float sky = smoothstep(0.7 * SKY_DISTANCE, SKY_DISTANCE, -view_z(uv));
    vec2 to_sun = (uv - u_sun_uv) * vec2(u_aspect, 1.0);
    return sky * exp(-GLOW_SHARPNESS * dot(to_sun, to_sun));
}

void main() {
    vec2 delta = (u_sun_uv - v_uv) * (MARCH_LENGTH / float(NB_STEPS));

    vec2 uv = v_uv;
    float light = 0.0;
    float weight = 1.0;
    for (int i = 0; i < NB_STEPS; i++) {
        light += weight * sky_glow(uv);
        weight *= DECAY;
        uv += delta;
    }

    // normalized by the total decay weight so the output stays in [0, 1]
    float total_weight = (1.0 - pow(DECAY, float(NB_STEPS))) / (1.0 - DECAY);
    fragColor = vec4(vec3(light / total_weight), 1.0);
}
