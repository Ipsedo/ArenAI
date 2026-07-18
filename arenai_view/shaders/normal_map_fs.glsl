#version 450

// per-frame globals, bound once per renderer frame (set 0 of every scene pipeline)
layout(set = 0, binding = 0, std140) uniform FrameGlobals {
    vec4 u_light_pos;// light position in view space (w unused)
    vec4 u_world_up; // world up axis in view space (xyz) and camera world height (w)
    vec4 u_fog_color;// rgb only
};

layout(set = 1, binding = 0) uniform sampler2D u_tex;
layout(set = 1, binding = 1) uniform sampler2D u_normal_map;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec2 v_tex_coord;
layout(location = 2) in vec3 v_normal;

layout(location = 0) out vec4 fragColor;

// https://geeks3d.developpez.com/normal-mapping-glsl/
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    // recupere les vecteurs du triangle composant le pixel ; dFdy est negate
    // pour retrouver l'orientation d'ecran GL (y vers le haut), sans quoi le
    // repere cofacteur (qui omet le signe du determinant) serait inverse
    vec3 dp1 = dFdx( p );
    vec3 dp2 = -dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = -dFdy( uv );

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
    vec3 map = texture(u_normal_map, texcoord).rgb;
    map = map * 255./127. - 128./127.;
    mat3 TBN = cotangent_frame(N, -V, texcoord);
    return normalize(TBN * map);
}

void main() {
    vec2 uv = v_tex_coord;

    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos.xyz - v_position);
    vec3 V = normalize(-v_position);
    vec3 PN = get_normal(N, V, uv);

    float lambertTerm = max(dot(PN, L), 0.0);
    fragColor = texture(u_tex, uv) * lambertTerm;
}
