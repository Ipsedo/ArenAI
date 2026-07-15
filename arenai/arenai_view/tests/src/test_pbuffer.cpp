//
// Created by samuel on 26/06/2026.
//

#include <fstream>
#include <numeric>
#include <unordered_set>

#include <glm/ext/matrix_transform.hpp>
#include <nlohmann/json.hpp>

#include <arenai_view/backend.h>
#include <arenai_view/camera.h>
#include <arenai_view_tests/test_pbuffer.h>

#include "./utils/local_file_reader.h"

using namespace arenai;
using namespace arenai::view;

TEST_P(PBufferParam, TestPBuffer) {
    const auto [width, height] = GetParam();

    const auto backend = view::make_opengl_backend();

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

    // 1. First call initializes PBOs and returns a black warmup frame
    const auto [black_pixels] = buffer_renderer->draw_and_get_frame(model_matrices);

    ASSERT_EQ(black_pixels.size(), 3 * width * height);

    // attempt black image
    ASSERT_EQ(
        std::accumulate(
            black_pixels.begin(), black_pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // 2. First frame
    const auto [pixels] = buffer_renderer->draw_and_get_frame(model_matrices);

    ASSERT_EQ(pixels.size(), 3 * width * height);

    // detect if black image
    ASSERT_GT(
        std::accumulate(
            pixels.begin(), pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // Golden image tests
    const auto golden_image_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "resources" / "golden_images"
        / ("golden_cubemap_" + std::to_string(width) + "_" + std::to_string(height) + ".json");

#ifdef ARENAI_REGENERATE_GOLDEN_IMAGES
    // rebuild mode: always overwrite the golden below (see ARENAI_REGENERATE_GOLDEN_IMAGES)
    if (false) {
#else
    if (std::filesystem::exists(golden_image_path)) {
#endif
        std::ifstream input_file(golden_image_path);
        nlohmann::json golden_image_json;
        input_file >> golden_image_json;

        const auto golden_pixels = golden_image_json.get<std::vector<uint8_t>>();

        for (size_t i = 0; i < golden_pixels.size(); ++i) {
            constexpr int tolerance = 2;
            ASSERT_LE(std::abs(golden_pixels[i] - pixels[i]), tolerance);
        }
    } else {
        // generate golden image for first run
        nlohmann::json output_json(pixels);
        std::ofstream output_file(golden_image_path);
        output_file << output_json;
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestPBuffer, PBufferParam,
    testing::Values(image_size(16, 16), image_size(16, 32), image_size(32, 32)));
