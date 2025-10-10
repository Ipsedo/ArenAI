#version 330

#extension GL_OES_standard_derivatives : enable
precision mediump float;
uniform sampler2D u_tex;
uniform sampler2D u_normal_map;
uniform vec3 u_light_pos;

varying vec3 v_position;
varying vec2 v_tex_coord;
varying vec3 v_normal;

// https://geeks3d.developpez.com/normal-mapping-glsl/
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    // recupere les vecteurs du triangle composant le pixel
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // resout le systeme lineaire
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construit une trame invariante a l'echelle
    float invmax = inversesqrt(max(dot(T,T), dot(B,B)));
    return mat3(T * invmax, B * invmax, N);
}

vec3 get_normal( vec3 N, vec3 V, vec2 texcoord ) {
    // N, la normale interpolee et
    // V, le vecteur vue (vertex dirige vers l'oeil)
    vec3 map = texture2D(u_normal_map, texcoord).rgb;
    map = map * 255./127. - 128./127.;
    mat3 TBN = cotangent_frame(N, -V, texcoord);
    return normalize(TBN * map);
}

void main() {
    vec2 uv = v_tex_coord;

    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos - v_position);
    vec3 V = normalize(-v_position);
    vec3 PN = get_normal(N, V, uv);

    float lambertTerm = max(dot(PN, L), 0.0);
    /*if (lambertTerm > 0.0) {
        final_color = lambertTerm * texColor;

        vec3 E = normalize(Vertex_EyeVec.xyz);
        vec3 R = reflect(-L, PN);
        float specular = pow( max(dot(R, E), 0.0), material_shininess);
        final_color += light_specular * material_specular * specular;
    }*/
    gl_FragColor = texture2D(u_tex, uv) * lambertTerm;
}
