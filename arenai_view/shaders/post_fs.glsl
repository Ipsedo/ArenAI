#version 450

layout(set = 1, binding = 0) uniform sampler2D u_scene;
layout(set = 1, binding = 1) uniform sampler2D u_ao;
layout(set = 1, binding = 2) uniform sampler2D u_bloom;
layout(set = 1, binding = 3) uniform sampler2D u_rays;

layout(push_constant) uniform Push {
    // god-rays strength, faded on the CPU when the sun leaves the frame
    float u_ray_strength;
    // frame counter animating the film grain
    float u_frame;
};

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 fragColor;

// exposure applied in linear space, before tonemapping; the ACES curve lifts
// mid-tones in the LDR range, so a value below 1 keeps the pass roughly
// brightness-neutral (only the gentle filmic S-curve remains)
const float EXPOSURE = 0.8;

// color grading: gentle saturation boost, plus split toning pushing shadows
// toward a cool blue and highlights toward a warm yellow
const float SATURATION = 1.12;
const vec3 SHADOW_TONE = vec3(0.94, 0.98, 1.08);
const vec3 HIGHLIGHT_TONE = vec3(1.06, 1.01, 0.94);

// radial RGB split, growing quadratically toward the corners
const float ABERRATION = 0.004;

const float BLOOM_STRENGTH = 0.6;
const vec3 SUN_COLOR = vec3(1.0, 0.92, 0.75);

const float VIGNETTE = 0.22;
// animated film grain; also acts as dithering, breaking the banding of the
// smooth fog and sky gradients
const float GRAIN = 0.016;

// ACES filmic tonemapping (Narkowicz fit)
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// sin-free hash: the classic fract(sin(dot)) collapses into visible diagonal
// stripes once its argument gets large (animated grain offsets)
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// the scene is rendered sRGB-encoded: decode to linear before any math
vec3 scene_linear(vec2 uv) { return pow(texture(u_scene, uv).rgb, vec3(2.2)); }

void main() {
    vec2 from_center = v_uv - 0.5;

    // chromatic aberration: red pushed outward, blue inward
    vec2 shift = from_center * dot(from_center, from_center) * ABERRATION;
    vec3 color = vec3(
        scene_linear(v_uv + shift).r, scene_linear(v_uv).g, scene_linear(v_uv - shift).b);

    // half-res ambient occlusion (already faded by fog in the SSAO pass)
    color *= texture(u_ao, v_uv).r;

    // bloom and volumetric sun rays, both accumulated in linear space
    color += BLOOM_STRENGTH * texture(u_bloom, v_uv).rgb;
    color += u_ray_strength * texture(u_rays, v_uv).r * SUN_COLOR;

    color = aces(color * EXPOSURE);

    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luma), color, SATURATION);
    color *= mix(SHADOW_TONE, HIGHLIGHT_TONE, luma);

    color *= 1.0 - VIGNETTE * smoothstep(0.35, 0.72, length(from_center));

    // encode back to sRGB for the display, then grain on the encoded value
    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
    float noise = hash(gl_FragCoord.xy + vec2(u_frame * 13.37, u_frame * 7.13));
    color += (noise - 0.5) * GRAIN;

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
