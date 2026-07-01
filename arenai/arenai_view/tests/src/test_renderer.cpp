//
// Created by claude on 01/07/2026.
//

#include <numeric>

#include <glm/ext/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <arenai_view/camera.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/pbuffer_renderer.h>
#include <arenai_view/specular.h>

#include "./utils/local_file_reader.h"
#include "./utils/local_gl_context.h"
#include "./utils/make_shapes.h"

class RendererTest : public testing::Test {
protected:
    void SetUp() override {
        gl_context = std::make_shared<LocalGlContext>();
        file_reader = std::make_shared<LocalAssetFileReader>(
            std::filesystem::path(__FILE__).parent_path() / ".." / ".." / ".." / ".." / "app"
            / "src" / "main" / "assets");
    }

    std::shared_ptr<LocalGlContext> gl_context;
    std::shared_ptr<LocalAssetFileReader> file_reader;
};

TEST_F(RendererTest, RemoveDrawableThenDrawClearColor) {
    constexpr int w = 16, h = 16;

    PBufferRenderer renderer(
        gl_context, w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    renderer.make_current();

    renderer.add_drawable("sky", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    const auto with_drawable = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

    // warmup + one real frame with the drawable
    renderer.draw_and_get_frame(with_drawable);
    const auto [with_pixels] = renderer.draw_and_get_frame(with_drawable);

    ASSERT_GT(
        std::accumulate(
            with_pixels.begin(), with_pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // remove the drawable
    renderer.remove_drawable("sky");

    const auto empty = std::vector<std::tuple<std::string, glm::mat4>>{};

    // draw two more frames (PBO double-buffering: need 2 to flush)
    renderer.draw_and_get_frame(empty);
    const auto [after_pixels] = renderer.draw_and_get_frame(empty);

    // should be clear color (red)
    const int hw = w * h;
    for (int i = 0; i < hw; ++i) {
        ASSERT_EQ(after_pixels[0 * hw + i], 255) << "R at pixel " << i;
        ASSERT_EQ(after_pixels[1 * hw + i], 0) << "G at pixel " << i;
        ASSERT_EQ(after_pixels[2 * hw + i], 0) << "B at pixel " << i;
    }
}

TEST_F(RendererTest, RemoveNonExistentDrawable) {
    constexpr int w = 16, h = 16;

    PBufferRenderer renderer(
        gl_context, w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    renderer.make_current();

    ASSERT_NO_THROW(renderer.remove_drawable("does_not_exist"));
}

TEST_F(RendererTest, MixedDrawablesCubeMapAndSpecular) {
    constexpr int w = 32, h = 32;

    PBufferRenderer renderer(
        gl_context, w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    renderer.make_current();

    // add cubemap alone, render
    renderer.add_drawable("sky", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    const auto sky_only_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

    renderer.draw_and_get_frame(sky_only_matrices);
    const auto [sky_pixels] = renderer.draw_and_get_frame(sky_only_matrices);

    // add specular cube on top
    auto [vertices, normals] = make_cube(2.f);
    renderer.add_drawable(
        "cube", std::make_unique<Specular>(
                    file_reader, vertices, normals, glm::vec4(0.2f, 0.2f, 0.2f, 1.f),
                    glm::vec4(0.f, 0.f, 1.f, 1.f), glm::vec4(1.f, 1.f, 1.f, 1.f), 32.f));

    const auto mixed_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}, {"cube", glm::mat4(1.f)}};

    renderer.draw_and_get_frame(mixed_matrices);
    const auto [mixed_pixels] = renderer.draw_and_get_frame(mixed_matrices);

    // mixed scene must differ from sky-only scene
    bool differs = false;
    for (size_t i = 0; i < sky_pixels.size(); ++i) {
        if (sky_pixels[i] != mixed_pixels[i]) {
            differs = true;
            break;
        }
    }
    ASSERT_TRUE(differs);
}

TEST_F(RendererTest, GetWidthHeight) {
    constexpr int w = 24, h = 48;

    PBufferRenderer renderer(
        gl_context, w, h, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    ASSERT_EQ(renderer.get_width(), w);
    ASSERT_EQ(renderer.get_height(), h);
}
