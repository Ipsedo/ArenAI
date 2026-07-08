//
// Created by claude on 01/07/2026.
//

#include <glm/ext/matrix_transform.hpp>

#include <arenai_view/camera.h>
#include <arenai_view/factory.h>
#include <arenai_view_tests/test_pbuffer.h>

#include "./utils/local_file_reader.h"

using namespace arenai;
using namespace arenai::view;

TEST_P(PBufferMultiFrameParam, StabilityMultiFrame) {
    const auto [width, height] = GetParam();

    const auto backend = view::make_opengl_view_factory()->make_backend();

    const auto buffer_renderer = backend->make_offscreen_renderer(
        width, height, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    buffer_renderer->make_current();

    buffer_renderer->add_drawable(
        "test", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

    const auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"test", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

    // warmup
    buffer_renderer->draw_and_get_frame(model_matrices);

    // reference frame
    const auto [ref_pixels] = buffer_renderer->draw_and_get_frame(model_matrices);
    ASSERT_EQ(ref_pixels.size(), 3 * width * height);

    // subsequent frames must be identical (static scene, same camera)
    constexpr int num_extra_frames = 8;
    for (int f = 0; f < num_extra_frames; ++f) {
        const auto [frame_pixels] = buffer_renderer->draw_and_get_frame(model_matrices);
        ASSERT_EQ(frame_pixels.size(), ref_pixels.size()) << "frame " << f;

        for (size_t i = 0; i < ref_pixels.size(); ++i) {
            ASSERT_EQ(frame_pixels[i], ref_pixels[i])
                << "frame " << f << " pixel diff at index " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestPBufferMultiFrame, PBufferMultiFrameParam,
    testing::Values(image_size(16, 16), image_size(16, 32), image_size(32, 32)));
