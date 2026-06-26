//
// Created by samuel on 26/06/2026.
//

#ifndef ARENAI_VIEW_TESTS_MAKE_SHAPES_H
#define ARENAI_VIEW_TESTS_MAKE_SHAPES_H
#include <tuple>
#include <vector>

std::tuple<
    std::vector<std::tuple<float, float, float>>, std::vector<std::tuple<float, float, float>>>
make_cube(float half_size);

#endif//ARENAI_VIEW_TESTS_MAKE_SHAPES_H
