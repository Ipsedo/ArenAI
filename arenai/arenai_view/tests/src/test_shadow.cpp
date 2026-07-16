//
// Created by samuel on 13/07/2026.
//

#include <numeric>

#include <glm/ext/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <arenai_view/camera.h>

#include "./utils/local_file_reader.h"
#include "./utils/make_shapes.h"
#include "opengl/drawables/diffuse.h"
#include "opengl/renderers/gl_offscreen_renderer.h"

using namespace arenai;
using namespace arenai::view;

namespace {

    constexpr int WIDTH = 64, HEIGHT = 64;

    // floor of 200x2x200 units centered at the origin
    const auto FLOOR_MATRIX = glm::scale(glm::mat4(1.f), glm::vec3(100.f, 1.f, 100.f));
    // cube of 40 units hanging at y=100, between the overhead light and the
    // floor, outside the camera frustum (only its shadow is visible)
    const auto OCCLUDER_MATRIX =
        glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(0.f, 100.f, 0.f)), glm::vec3(20.f));

    long pixel_sum(const std::vector<uint8_t> &pixels) {
        return std::accumulate(
            pixels.begin(), pixels.end(), 0L,
            [](const long acc, const uint8_t p) { return acc + static_cast<long>(p); });
    }

    std::unique_ptr<GlOffscreenRenderer>
    make_shadow_test_renderer(const bool with_shadows, long &floor_only_sum, long &occluded_sum) {
        const auto main_context = std::make_shared<HeadlessEglContext>();
        const auto file_reader =
            std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

        auto renderer = std::make_unique<GlOffscreenRenderer>(
            main_context, WIDTH, HEIGHT, glm::vec3(0.f, 300.f, 0.f),
            std::make_shared<StaticCamera>(
                glm::vec3{0.f, 80.f, -80.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}),
            with_shadows);

        renderer->make_current();

        auto [vertices, normals] = make_cube(1.f);
        const auto make_drawable = [&] {
            return std::make_unique<Diffuse>(
                file_reader, vertices, glm::vec4(0.8f, 0.8f, 0.8f, 1.f));
        };
        renderer->add_drawable("floor", make_drawable());
        renderer->add_drawable("occluder", make_drawable());

        const auto floor_only =
            std::vector<std::tuple<std::string, glm::mat4>>{{"floor", FLOOR_MATRIX}};
        const auto with_occluder = std::vector<std::tuple<std::string, glm::mat4>>{
            {"floor", FLOOR_MATRIX}, {"occluder", OCCLUDER_MATRIX}};

        // PBO double-buffering: draw twice, keep the second frame
        renderer->draw_and_get_frame(floor_only);
        floor_only_sum = pixel_sum(renderer->draw_and_get_frame(floor_only).pixels);

        renderer->draw_and_get_frame(with_occluder);
        occluded_sum = pixel_sum(renderer->draw_and_get_frame(with_occluder).pixels);

        return renderer;
    }

}// namespace

TEST(ShadowRenderingTest, OccluderDarkensFloor) {
    long floor_only_sum = 0, occluded_sum = 0;
    const auto renderer = make_shadow_test_renderer(true, floor_only_sum, occluded_sum);

    ASSERT_GT(floor_only_sum, 0);

    // the occluder is not visible from the camera, so any brightness drop
    // comes from its shadow on the floor
    constexpr long margin = 3L * WIDTH * HEIGHT;// per-pixel GL tolerance
    ASSERT_LT(occluded_sum + margin, floor_only_sum);
}

TEST(ShadowRenderingTest, NoShadowWhenDisabled) {
    long floor_only_sum = 0, occluded_sum = 0;
    const auto renderer = make_shadow_test_renderer(false, floor_only_sum, occluded_sum);

    ASSERT_GT(floor_only_sum, 0);

    // without shadows, the out-of-frustum occluder must not change the image
    constexpr long tolerance = 3L * WIDTH * HEIGHT;
    ASSERT_NEAR(occluded_sum, floor_only_sum, tolerance);
}
