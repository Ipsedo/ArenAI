//
// Created by samuel on 26/06/2026.
//

#include "./make_shapes.h"

std::tuple<
    std::vector<std::tuple<float, float, float>>, std::vector<std::tuple<float, float, float>>>
make_cube(const float half_size) {
    const float s = half_size;

    // Les 8 coins
    const std::tuple<float, float, float> corners[8] = {
        {s, s, s},   // 0
        {-s, s, s},  // 1
        {-s, -s, s}, // 2
        {s, -s, s},  // 3
        {s, s, -s},  // 4
        {-s, s, -s}, // 5
        {-s, -s, -s},// 6
        {s, -s, -s}, // 7
    };

    struct Face {
        int a, b, c, d;
        std::tuple<float, float, float> normal;
    };
    const Face faces[6] = {
        {.a = 0, .b = 1, .c = 2, .d = 3, .normal = {0.f, 0.f, 1.f}}, // avant   (+Z)
        {.a = 4, .b = 7, .c = 6, .d = 5, .normal = {0.f, 0.f, -1.f}},// arrière (-Z)
        {.a = 0, .b = 3, .c = 7, .d = 4, .normal = {1.f, 0.f, 0.f}}, // droite  (+X)
        {.a = 1, .b = 5, .c = 6, .d = 2, .normal = {-1.f, 0.f, 0.f}},// gauche  (-X)
        {.a = 0, .b = 4, .c = 5, .d = 1, .normal = {0.f, 1.f, 0.f}}, // haut    (+Y)
        {.a = 2, .b = 6, .c = 7, .d = 3, .normal = {0.f, -1.f, 0.f}},// bas     (-Y)
    };

    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;
    vertices.reserve(36);
    normals.reserve(36);

    for (const auto &[a, b, c, d, normal]: faces) {
        for (const int tri[6] = {a, b, c, a, c, d}; const int idx: tri) {
            vertices.push_back(corners[idx]);
            normals.push_back(normal);
        }
    }

    return {vertices, normals};
}
