#version 300 es

precision mediump float;

uniform sampler2D u_scene;

in vec2 v_uv;

out vec4 fragColor;

// soft luminance knee keeping only the brightest parts of the LDR scene
// (sun glints, explosions, tracers)
const float THRESHOLD = 0.72;
const float KNEE = 0.25;

void main() {
    // scene is sRGB-encoded: bloom is accumulated in linear space
    vec3 color = pow(texture(u_scene, v_uv).rgb, vec3(2.2));
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

    fragColor = vec4(color * smoothstep(THRESHOLD - KNEE, THRESHOLD + KNEE, luma), 1.0);
}
