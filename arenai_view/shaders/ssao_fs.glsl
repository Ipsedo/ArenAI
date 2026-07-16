#version 330 core

precision highp float;

uniform highp sampler2D u_depth;
// projection terms: xy = (proj[0][0], proj[1][1]), zw = (proj[2][2], proj[3][2])
uniform vec4 u_proj_info;

in vec2 v_uv;

out vec4 fragColor;

const int NB_TAPS = 10;
// world-space radius of the occlusion hemisphere
const float RADIUS = 4.0;
const float BIAS = 0.06;
const float INTENSITY = 0.9;
// AO fades out where the fog takes over (must match diffuse_shadow_fs.glsl)
const float FOG_DENSITY = 0.0014;
// beyond this view-space distance everything counts as sky/background
const float SKY_DISTANCE = 900.0;
const float GOLDEN_ANGLE = 2.39996;

float view_z(vec2 uv) {
    float ndc_z = texture(u_depth, uv).r * 2.0 - 1.0;
    return -u_proj_info.w / (ndc_z + u_proj_info.z);
}

vec3 view_pos(vec2 uv) {
    float z = view_z(uv);
    return vec3((uv * 2.0 - 1.0) * -z / u_proj_info.xy, z);
}

void main() {
    vec3 pos = view_pos(v_uv);

    if (-pos.z > SKY_DISTANCE) {
        fragColor = vec4(1.0);
        return;
    }

    // face normal reconstructed from the depth buffer alone
    vec3 normal = normalize(cross(dFdx(pos), dFdy(pos)));

    // screen-space (uv) radius of the world-space hemisphere at this depth
    float uv_radius = 0.5 * RADIUS * u_proj_info.y / -pos.z;

    // per-pixel random rotation of the spiral turns banding into noise
    // (sin-free hash: fract(sin(dot)) degrades into diagonal stripes at
    // large screen coordinates)
    vec3 p3 = fract(vec3(gl_FragCoord.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    float angle = 6.2831853 * fract((p3.x + p3.y) * p3.z);

    // Alchemy AO estimator over a spiral of taps
    float occlusion = 0.0;
    for (int i = 0; i < NB_TAPS; i++) {
        float r = sqrt((float(i) + 0.5) / float(NB_TAPS));
        float a = angle + float(i) * GOLDEN_ANGLE;
        vec3 tap = view_pos(v_uv + r * uv_radius * vec2(cos(a), sin(a)));

        vec3 v = tap - pos;
        occlusion += max(0.0, dot(v, normal) - BIAS) / (dot(v, v) + 0.05);
    }
    // the estimator has a 1/distance unit: the radius normalizes it
    float ao = clamp(1.0 - INTENSITY * RADIUS * occlusion / float(NB_TAPS), 0.0, 1.0);

    float fade = exp(-FOG_DENSITY * length(pos));
    fragColor = vec4(vec3(mix(1.0, ao, fade)), 1.0);
}
