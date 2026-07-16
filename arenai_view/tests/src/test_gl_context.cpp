//
// Created by claude on 01/07/2026.
//

#include <numeric>

#include <glm/ext/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <arenai_view/backend.h>
#include <arenai_view/camera.h>

#include "./utils/local_file_reader.h"

using namespace arenai;
using namespace arenai::view;

TEST(GLContextTest, MultipleRenderersSameContext) {
    const auto backend = view::make_opengl_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 16, h = 16;

    const auto renderer_a = backend->make_offscreen_renderer(
        w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    const auto renderer_b = backend->make_offscreen_renderer(
        w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    // renderer_a with cubemap
    renderer_a->make_current();
    renderer_a->add_drawable(
        "sky", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

    // renderer_b stays empty (clear color only)
    renderer_b->make_current();

    const auto sky_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};
    const auto empty_matrices = std::vector<std::tuple<std::string, glm::mat4>>{};

    // draw renderer_a
    renderer_a->make_current();
    renderer_a->draw_and_get_frame(sky_matrices);
    const auto [pixels_a] = renderer_a->draw_and_get_frame(sky_matrices);

    // draw renderer_b
    renderer_b->make_current();
    renderer_b->draw_and_get_frame(empty_matrices);
    const auto [pixels_b] = renderer_b->draw_and_get_frame(empty_matrices);

    ASSERT_EQ(pixels_a.size(), pixels_b.size());

    // renderer_a should have cubemap content (not just clear color)
    ASSERT_GT(
        std::accumulate(
            pixels_a.begin(), pixels_a.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // they should differ (cubemap vs clear color)
    bool differs = false;
    for (size_t i = 0; i < pixels_a.size(); ++i) {
        if (pixels_a[i] != pixels_b[i]) {
            differs = true;
            break;
        }
    }
    ASSERT_TRUE(differs);
}

TEST(GLContextTest, MakeCurrentReleaseCurrent) {
    const auto backend = view::make_opengl_backend();

    constexpr int w = 16, h = 16;
    const auto renderer = backend->make_offscreen_renderer(
        w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    ASSERT_NO_THROW(renderer->make_current());
    ASSERT_NO_THROW(renderer->release_current());
    ASSERT_NO_THROW(renderer->make_current());
    ASSERT_NO_THROW(renderer->release_current());
}

TEST(GLContextTest, AlternateContexts) {
    const auto backend = view::make_opengl_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 16, h = 16;

    const auto renderer_a = backend->make_offscreen_renderer(
        w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    const auto renderer_b = backend->make_offscreen_renderer(
        w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    renderer_a->make_current();
    renderer_a->add_drawable(
        "sky", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

    const auto sky_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};
    const auto empty_matrices = std::vector<std::tuple<std::string, glm::mat4>>{};

    // alternate between contexts multiple times
    for (int i = 0; i < 4; ++i) {
        renderer_a->make_current();
        ASSERT_NO_THROW(renderer_a->draw_and_get_frame(sky_matrices));

        renderer_b->make_current();
        ASSERT_NO_THROW(renderer_b->draw_and_get_frame(empty_matrices));
    }
}
