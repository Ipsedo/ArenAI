#version 150

precision mediump float;

varying vec3 v_tex_coords;
uniform samplerCube u_cube_map;

void main () {
  gl_FragColor = textureCube(u_cube_map, v_tex_coords);
}