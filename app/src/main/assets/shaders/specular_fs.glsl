#version 300 es

precision highp float;
precision highp int;

uniform vec3 u_cam_pos;
uniform vec3 u_light_pos;

uniform vec4 u_ambient_color;
uniform vec4 u_diffuse_color;
uniform vec4 u_specular_color;
uniform float u_shininess;

in vec3 v_position;
in vec3 v_normal;

out vec4 fragColor;

void main() {
    float distance      = length(u_light_pos - v_position);
    vec3  light_vector  = normalize(u_light_pos - v_position);

    float diffuse_coeff = max(dot(v_normal, light_vector), 0.1);
    vec4  diffuse       = diffuse_coeff * u_diffuse_color;

    float specular_coeff = 0.0;
    if (diffuse_coeff > 0.0) {
        vec3 incidence_vector   = -light_vector;
        vec3 reflection_vector  = reflect(incidence_vector, v_normal);
        vec3 surface_to_camera  = normalize(u_cam_pos - v_position);
        float cos_angle         = max(0.0, dot(surface_to_camera, reflection_vector));
        specular_coeff          = pow(cos_angle, u_shininess);
    }

    vec4 specular = specular_coeff * u_specular_color;
    vec4 ambient  = 0.1 * u_ambient_color;

    fragColor = ambient + diffuse + specular;
}
