//
// Created by claude on 01/07/2026.
//

#include <cmath>

#include <gtest/gtest.h>

#include "./utils/make_shapes.h"

struct cube_param {
    float half_size;
};

class MakeCubeParam : public testing::TestWithParam<cube_param> {};

TEST_P(MakeCubeParam, VertexCount) {
    const auto [half_size] = GetParam();
    auto [vertices, normals] = make_cube(half_size);

    // 6 faces * 2 triangles * 3 vertices = 36
    ASSERT_EQ(vertices.size(), 36);
    ASSERT_EQ(normals.size(), 36);
}

TEST_P(MakeCubeParam, VerticesInBoundingBox) {
    const auto [half_size] = GetParam();
    auto [vertices, normals] = make_cube(half_size);

    for (size_t i = 0; i < vertices.size(); ++i) {
        auto [x, y, z] = vertices[i];
        ASSERT_LE(std::abs(x), half_size + 1e-6f) << "vertex " << i;
        ASSERT_LE(std::abs(y), half_size + 1e-6f) << "vertex " << i;
        ASSERT_LE(std::abs(z), half_size + 1e-6f) << "vertex " << i;
    }
}

TEST_P(MakeCubeParam, NormalsAreUnit) {
    const auto [half_size] = GetParam();
    auto [vertices, normals] = make_cube(half_size);

    for (size_t i = 0; i < normals.size(); ++i) {
        auto [nx, ny, nz] = normals[i];
        const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        ASSERT_NEAR(length, 1.f, 1e-5f) << "normal " << i;
    }
}

TEST_P(MakeCubeParam, NormalsAreAxisAligned) {
    const auto [half_size] = GetParam();
    auto [vertices, normals] = make_cube(half_size);

    for (size_t i = 0; i < normals.size(); ++i) {
        auto [nx, ny, nz] = normals[i];

        // each normal should have exactly one non-zero component
        int non_zero = 0;
        if (std::abs(nx) > 1e-6f) non_zero++;
        if (std::abs(ny) > 1e-6f) non_zero++;
        if (std::abs(nz) > 1e-6f) non_zero++;

        ASSERT_EQ(non_zero, 1) << "normal " << i << ": (" << nx << ", " << ny << ", " << nz << ")";
    }
}

TEST_P(MakeCubeParam, SixDistinctNormals) {
    const auto [half_size] = GetParam();
    auto [vertices, normals] = make_cube(half_size);

    std::set<std::tuple<int, int, int>> unique_normals;
    for (const auto &[nx, ny, nz]: normals) {
        unique_normals.insert(
            {static_cast<int>(std::round(nx)), static_cast<int>(std::round(ny)),
             static_cast<int>(std::round(nz))});
    }

    ASSERT_EQ(unique_normals.size(), 6);
}

INSTANTIATE_TEST_SUITE_P(
    TestMakeCube, MakeCubeParam,
    testing::Values(cube_param{0.5f}, cube_param{1.f}, cube_param{10.f}));
