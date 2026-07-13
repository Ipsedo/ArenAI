//
// Created by claude on 01/07/2026.
//

#include <fstream>
#include <numeric>

#include <glm/ext/matrix_transform.hpp>
#include <nlohmann/json.hpp>

#include <arenai_view/backend.h>
#include <arenai_view/camera.h>
#include <arenai_view_tests/test_pbuffer.h>

#include "./utils/local_file_reader.h"
#include "./utils/make_shapes.h"

using namespace arenai;
using namespace arenai::view;

TEST_P(PBufferDiffuseParam, TestDiffuseRendering) {
    const auto [width, height] = GetParam();

    const auto backend = view::make_opengl_backend();

    const auto buffer_renderer = backend->make_offscreen_renderer(
        width, height, {0.f, 10.f, -2.f},
        std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));

    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    buffer_renderer->make_current();

    auto [vertices, normals] = make_cube(2.f);

    buffer_renderer->add_drawable(
        "cube", backend->drawable_factory()->make_diffuse(
                    file_reader, vertices, glm::vec4(0.f, 0.f, 1.f, 1.f)));

    const auto model_matrices =
        std::vector<std::tuple<std::string, glm::mat4>>{{"cube", glm::mat4(1.f)}};

    // warmup
    const auto [black_pixels] = buffer_renderer->draw_and_get_frame(model_matrices);
    ASSERT_EQ(black_pixels.size(), 3 * width * height);
    ASSERT_EQ(
        std::accumulate(
            black_pixels.begin(), black_pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // actual frame
    const auto [pixels] = buffer_renderer->draw_and_get_frame(model_matrices);
    ASSERT_EQ(pixels.size(), 3 * width * height);

    ASSERT_GT(
        std::accumulate(
            pixels.begin(), pixels.end(), 0,
            [](const int acc, const uint8_t p) { return static_cast<int>(p) + acc; }),
        0);

    // golden image
    const auto golden_image_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "resources" / "golden_images"
        / ("golden_diffuse_" + std::to_string(width) + "_" + std::to_string(height) + ".json");

    if (std::filesystem::exists(golden_image_path)) {
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
    TestPBufferDiffuse, PBufferDiffuseParam,
    testing::Values(image_size(16, 16), image_size(16, 32), image_size(32, 32)));
