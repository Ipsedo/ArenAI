uniform mat4 u_mvp_matrix;
attribute vec4 a_position;

void main() {
    gl_Position = u_mvp_matrix * a_position;
}