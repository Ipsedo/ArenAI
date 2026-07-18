#version 450

layout(set = 1, binding = 0, std140) uniform Material { vec4 u_color; };

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = u_color;
}
