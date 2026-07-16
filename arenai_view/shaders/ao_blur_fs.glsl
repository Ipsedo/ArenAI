#version 330 core

precision mediump float;

uniform sampler2D u_ao;

in vec2 v_uv;

out vec4 fragColor;

// 5x5 gaussian at half resolution, hiding the SSAO's rotated-spiral noise
const float WEIGHTS[3] = float[](6.0, 4.0, 1.0);

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_ao, 0));

    float sum = 0.0;
    for (int y = -2; y <= 2; y++)
        for (int x = -2; x <= 2; x++) {
            float weight = WEIGHTS[abs(x)] * WEIGHTS[abs(y)];
            sum += weight * texture(u_ao, v_uv + vec2(x, y) * texel).r;
        }

    fragColor = vec4(vec3(sum / 256.0), 1.0);
}
