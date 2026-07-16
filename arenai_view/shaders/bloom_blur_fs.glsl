#version 330 core

precision mediump float;

uniform sampler2D u_source;
// (1, 0) for the horizontal pass, (0, 1) for the vertical one
uniform vec2 u_direction;

in vec2 v_uv;

out vec4 fragColor;

// 9-tap separable gaussian, offsets stretched 1.5x for a wider glow
const float WEIGHTS[5] =
    float[](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);

void main() {
    vec2 step_uv = u_direction / vec2(textureSize(u_source, 0));

    vec3 sum = WEIGHTS[0] * texture(u_source, v_uv).rgb;
    for (int i = 1; i < 5; i++) {
        vec2 offset = 1.5 * float(i) * step_uv;
        sum += WEIGHTS[i]
               * (texture(u_source, v_uv + offset).rgb + texture(u_source, v_uv - offset).rgb);
    }

    fragColor = vec4(sum, 1.0);
}
