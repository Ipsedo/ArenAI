//
// Created by claude on 01/07/2026.
//

#include <numeric>

#include <arenai_view/camera.h>
#include <arenai_view/factory.h>
#include <arenai_view_tests/test_pbuffer.h>

using namespace arenai;
using namespace arenai::view;

TEST_P(PBufferClearColorParam, ClearColorNoDrawable) {
    const auto [width, height] = GetParam();

    const auto backend = view::make_opengl_backend();

    const auto buffer_renderer = backend->make_offscreen_renderer(
        width, height, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    buffer_renderer->make_current();

    constexpr auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>{};

    // warmup frame (black)
    buffer_renderer->draw_and_get_frame(model_matrices);

    // actual frame: should be the clear color (red = 1,0,0,0 in on_new_frame)
    const auto [pixels] = buffer_renderer->draw_and_get_frame(model_matrices);

    const int hw = width * height;
    ASSERT_EQ(pixels.size(), 3 * hw);

    // CHW layout: [R plane | G plane | B plane]
    // clear color is (1.0, 0.0, 0.0) => R=255, G=0, B=0
    for (int i = 0; i < hw; ++i) {
        ASSERT_EQ(pixels[0 * hw + i], 255) << "R channel pixel " << i;
        ASSERT_EQ(pixels[1 * hw + i], 0) << "G channel pixel " << i;
        ASSERT_EQ(pixels[2 * hw + i], 0) << "B channel pixel " << i;
    }
}

TEST_P(PBufferClearColorParam, PixelLayoutCHW) {
    const auto [width, height] = GetParam();

    const auto backend = view::make_opengl_backend();

    const auto buffer_renderer = backend->make_offscreen_renderer(
        width, height, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    buffer_renderer->make_current();

    constexpr auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>{};

    buffer_renderer->draw_and_get_frame(model_matrices);
    const auto [pixels] = buffer_renderer->draw_and_get_frame(model_matrices);

    const int hw = width * height;

    // Verify pixel (x, y) at channel c is at index c * hw + y * width + x
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            // All red clear color: R=255 at channel 0, G=0 at channel 1, B=0 at channel 2
            ASSERT_EQ(pixels[0 * hw + idx], 255) << "R at (" << x << ", " << y << ")";
            ASSERT_EQ(pixels[1 * hw + idx], 0) << "G at (" << x << ", " << y << ")";
            ASSERT_EQ(pixels[2 * hw + idx], 0) << "B at (" << x << ", " << y << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestPBufferClearColor, PBufferClearColorParam,
    testing::Values(
        image_size(16, 16), image_size(16, 32), image_size(32, 32), image_size(3, 7),
        image_size(1, 64), image_size(64, 1)));
