#version 300 es

precision highp float;
precision highp int;

uniform vec3 u_cam_pos;
uniform vec3 u_light_pos;

uniform vec4 u_ambient_color;
uniform vec4 u_diffuse_color;
uniform vec4 u_specular_color;
uniform float u_shininess;

uniform highp sampler2DShadow u_shadow_map;

in vec3 v_position;
in vec3 v_normal;
in vec4 v_shadow_coord;

out vec4 fragColor;

const vec2 POISSON[12] = vec2[](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457),
    vec2(-0.203, 0.621), vec2(0.962, -0.195), vec2(0.473, -0.480),
    vec2(0.519, 0.767), vec2(0.185, -0.893), vec2(0.507, 0.064),
    vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

// fraction of the diffuse light kept in fully shadowed areas, so that the
// object's own color still shows through instead of going black
const float SHADOW_MIN_LIGHT = 0.45;

float shadow_factor() {
    vec3 coord = v_shadow_coord.xyz / v_shadow_coord.w;

    // outside the shadow frustum: consider the fragment fully lit
    if (coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0 || coord.z > 1.0)
        return 1.0;

    // constant bias; slope-scaled acne removal is handled by glPolygonOffset
    // during the depth pass
    const float bias = 0.0005;
    const float radius = 3.5;
    vec2 texel_size = 1.0 / vec2(textureSize(u_shadow_map, 0));

    // random per-pixel rotation of the Poisson disk turns banding into noise
    float angle =
        6.2831853 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float c = cos(angle), s = sin(angle);
    mat2 rotation = mat2(c, s, -s, c);

    float lit = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 offset = rotation * POISSON[i] * radius * texel_size;
        lit += texture(u_shadow_map, vec3(coord.xy + offset, coord.z - bias));
    }
    return lit / 12.0;
}

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

    float shadow = shadow_factor();

    // shadows attenuate the diffuse term instead of removing it; specular
    // highlights are fully cut (a shiny glint inside a shadow looks wrong)
    float diffuse_shadow = mix(SHADOW_MIN_LIGHT, 1.0, shadow);

    fragColor = ambient + diffuse_shadow * diffuse + shadow * specular;
}
