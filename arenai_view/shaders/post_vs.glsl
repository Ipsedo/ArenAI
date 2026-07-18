#version 450

layout(location = 0) out vec2 v_uv;

// buffer-less fullscreen triangle: 3 vertices covering the whole viewport,
// uv derived from gl_VertexIndex (no vertex buffer to bind)
void main() {
    vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
