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
        {0, 1, 2, 3, {0.f, 0.f, 1.f}}, // avant   (+Z)
        {4, 7, 6, 5, {0.f, 0.f, -1.f}},// arrière (-Z)
        {0, 3, 7, 4, {1.f, 0.f, 0.f}}, // droite  (+X)
        {1, 5, 6, 2, {-1.f, 0.f, 0.f}},// gauche  (-X)
        {0, 4, 5, 1, {0.f, 1.f, 0.f}}, // haut    (+Y)
        {2, 6, 7, 3, {0.f, -1.f, 0.f}},// bas     (-Y)
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
