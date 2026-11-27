#include <fstream>

#include <glm/ext/matrix_transform.hpp>
#include <nlohmann/json.hpp>

#include <arenai_view/backend.h>
#include <arenai_view/camera.h>
#include <arenai_view_tests/test_offscreen.h>

#include "./utils/make_shapes.h"

using namespace arenai;
using namespace arenai::view;

namespace {
    // RGB sum of one pixel of a [C, H, W] square frame
    int brightness(const std::vector<uint8_t> &pixels, const int size, const int x, const int y) {
        const int plane = size * size, offset = y * size + x;
        return pixels[offset] + pixels[plane + offset] + pixels[2 * plane + offset];
    }

    std::vector<uint8_t> render_cube(const int width, const int height, const bool hatched) {
        const auto backend = view::make_vulkan_backend();
        const auto renderer = backend->make_offscreen_renderer(
            width, height, {0.f, 10.f, -2.f},
            std::make_shared<StaticCamera>(
                glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f}));
        renderer->make_current();

        auto [vertices, normals] = make_cube(2.f);
        const auto factory = backend->drawable_factory();
        const auto mesh = factory->make_mesh(vertices);
        constexpr glm::vec4 color(0.9f, 0.6f, 0.3f, 1.f);
        renderer->add_drawable(
            "cube",
            hatched ? factory->make_hatched(mesh, color) : factory->make_diffuse(mesh, color));

        // the cube fills most of the frame
        const auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
            {"cube", glm::scale(glm::mat4(1.f), glm::vec3(2.f))}};
        return renderer->draw_and_get_frame(model_matrices).pixels;
    }
}// namespace

TEST(OffscreenHatched, BandsFollowTheHeightPeriod) {
    constexpr int size = 128;
    // hatch period in pixels: render height / 32
    constexpr int period = size / 32;

    const auto plain = render_cube(size, size, false);
    const auto hatched = render_cube(size, size, true);

    // dark[y][x]: the hatched pixel is a band of the plain one
    std::vector dark(size, std::vector(size, false));
    std::vector covered(size, std::vector(size, false));
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            const int base = brightness(plain, size, x, y);
            const int shade = brightness(hatched, size, x, y);
            ASSERT_LE(shade, base + 3);
            covered[y][x] = base > 60;
            dark[y][x] = 10 * shade < 7 * base;
        }

    int interior = 0, nb_dark = 0, along_diagonal = 0, flipped_half_period = 0;
    for (int y = period; y < size - period; y++)
        for (int x = period; x < size - period; x++) {
            bool inside = true;
            for (int dy = -period; dy <= period; dy++)
                for (int dx = -period; dx <= period; dx++) inside &= covered[y + dy][x + dx];
            if (!inside) continue;

            interior++;
            nb_dark += dark[y][x];
            // a band is a diagonal line; either diagonal, the readback may flip rows
            along_diagonal += dark[y][x] == dark[y + 1][x - 1] || dark[y][x] == dark[y + 1][x + 1];
            flipped_half_period += dark[y][x] != dark[y][x + period / 2];
        }

    ASSERT_GT(interior, size * size / 4);
    EXPECT_NEAR(static_cast<float>(nb_dark) / static_cast<float>(interior), 0.5f, 0.05f);
    EXPECT_GE(static_cast<float>(along_diagonal) / static_cast<float>(interior), 0.98f);
    EXPECT_GE(static_cast<float>(flipped_half_period) / static_cast<float>(interior), 0.98f);
}

TEST_P(OffscreenHatchedParam, TestHatchedGoldenImage) {
    const auto [width, height] = GetParam();
    const auto pixels = render_cube(width, height, true);
    ASSERT_EQ(pixels.size(), 3 * width * height);

#ifndef ARENAI_REGENERATE_GOLDEN_IMAGES
    ARENAI_SKIP_UNLESS_PINNED_RENDER_ENV();
#endif

    const auto golden_image_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "resources" / "golden_images"
        / ("golden_hatched_" + std::to_string(width) + "_" + std::to_string(height) + ".json");

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
    TestOffscreenHatched, OffscreenHatchedParam, testing::Values(image_size(64, 64)));
