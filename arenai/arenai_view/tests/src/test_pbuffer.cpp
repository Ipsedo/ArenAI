//
// Created by samuel on 26/06/2026.
//

#include <numeric>
#include <unordered_set>

#include <glm/ext/matrix_transform.hpp>

#include <arenai_view/camera.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/pbuffer_renderer.h>
#include <arenai_view_tests/test_pbuffer.h>

#include "./local_file_reader.h"
#include "./local_gl_context.h"
#include "./make_shapes.h"

TEST_P(PBufferParam, TestPBuffer) {
    const auto [width, height] = GetParam();

    const auto local_gl_context = std::make_shared<LocalGlContext>();

    PBufferRenderer buffer_renderer(
        local_gl_context, width, height, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    const auto file_reader = std::make_shared<LocalAssetFileReader>(
        std::filesystem::path(__FILE__).parent_path() / ".." / ".." / ".." / ".." / "app" / "src"
        / "main" / "assets");

    buffer_renderer.make_current();

    buffer_renderer.add_drawable("test", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    const auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"test", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

    // First call initializes PBOs and returns a black warmup frame
    const auto _ = buffer_renderer.draw_and_get_frame(model_matrices);
    const auto [pixels] = buffer_renderer.draw_and_get_frame(model_matrices);

    ASSERT_EQ(pixels.size(), 3 * width * height);

    // detect if black image
    ASSERT_GT(
        std::accumulate(
            pixels.begin(), pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    std::unordered_set<uint8_t> pixels_set;

    for (const auto &p: pixels) pixels_set.emplace(p);

    ASSERT_GT(pixels_set.size(), 3);// detect if uniform image, 3 different color at minimum
}

INSTANTIATE_TEST_SUITE_P(
    TestPBuffer, PBufferParam,
    testing::Values(image_size(16, 16), image_size(16, 32), image_size(32, 32)));
