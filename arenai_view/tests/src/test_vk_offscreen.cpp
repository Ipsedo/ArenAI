//
// Created by samuel on 17/07/2026.
//

#include <cstdlib>
#include <numeric>
#include <thread>

#include <glm/ext/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <arenai_view/backend.h>
#include <arenai_view/camera.h>

#include "./utils/local_file_reader.h"
#include "vulkan/scene/renderers/offscreen_renderer.h"
#include "vulkan/vulkan_backend.h"

using namespace arenai;
using namespace arenai::view;

namespace {

    int pixel_sum(const std::vector<uint8_t> &pixels) {
        return std::accumulate(pixels.begin(), pixels.end(), 0, [](const int acc, const uint8_t p) {
            return static_cast<int>(p) + acc;
        });
    }

    std::shared_ptr<StaticCamera> make_camera() {
        return std::make_shared<StaticCamera>(
            glm::vec3{0.f, 0.f, -10.f}, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0.f, 1.f, 0.f});
    }

}// namespace

TEST(VkOffscreenTest, BackendInfo) {
    const auto backend = make_vulkan_backend();
    ASSERT_FALSE(backend->renderer_info().empty());
    ASSERT_NO_THROW(backend->release_thread());
}

TEST(VkOffscreenTest, WarmupThenCubemap) {
    const auto backend = make_vulkan_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 16, h = 16;
    const auto renderer = backend->make_offscreen_renderer(w, h, {0.f, 10.f, -2.f}, make_camera());

    renderer->make_current();
    renderer->add_drawable(
        "sky", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

    const auto matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

    // first call: black warm-up frame (2-frame readback latency)
    const auto [black_pixels] = renderer->draw_and_get_frame(matrices);
    ASSERT_EQ(black_pixels.size(), 3 * w * h);
    ASSERT_EQ(pixel_sum(black_pixels), 0);

    // second call: the first real frame
    const auto [pixels] = renderer->draw_and_get_frame(matrices);
    ASSERT_EQ(pixels.size(), 3 * w * h);
    ASSERT_GT(pixel_sum(pixels), 0);
}

TEST(VkOffscreenTest, DiffuseNonBlack) {
    const auto backend = make_vulkan_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 32, h = 32;
    const auto renderer = backend->make_offscreen_renderer(w, h, {0.f, 10.f, -2.f}, make_camera());

    // camera at -z looking at the origin: a big triangle facing it
    const std::vector<std::tuple<float, float, float>> vertices{
        {-5.f, -5.f, 0.f}, {0.f, 5.f, 0.f}, {5.f, -5.f, 0.f}};
    renderer->add_drawable(
        "tri", backend->drawable_factory()->make_diffuse(
                   file_reader, vertices, glm::vec4(0.f, 1.f, 0.f, 1.f)));

    const auto matrices = std::vector<std::tuple<std::string, glm::mat4>>{{"tri", glm::mat4(1.f)}};

    renderer->draw_and_get_frame(matrices);
    const auto [pixels] = renderer->draw_and_get_frame(matrices);

    // the green channel must dominate somewhere: the triangle is lit green
    const int hw = w * h;
    const int green = std::accumulate(pixels.begin() + hw, pixels.begin() + 2 * hw, 0);
    ASSERT_GT(green, 0);
}

TEST(VkOffscreenTest, TwoRenderersAlternate) {
    const auto backend = make_vulkan_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 16, h = 16;
    const auto renderer_a =
        backend->make_offscreen_renderer(w, h, {0.f, 10.f, -2.f}, make_camera());
    const auto renderer_b =
        backend->make_offscreen_renderer(w, h, {0.f, 10.f, -2.f}, make_camera());

    renderer_a->add_drawable(
        "sky", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

    const auto sky_matrices = std::vector<std::tuple<std::string, glm::mat4>>{
        {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};
    const auto empty_matrices = std::vector<std::tuple<std::string, glm::mat4>>{};

    for (int i = 0; i < 4; ++i) {
        renderer_a->make_current();
        ASSERT_NO_THROW(renderer_a->draw_and_get_frame(sky_matrices));

        renderer_b->make_current();
        ASSERT_NO_THROW(renderer_b->draw_and_get_frame(empty_matrices));
    }

    const auto [pixels_a] = renderer_a->draw_and_get_frame(sky_matrices);
    const auto [pixels_b] = renderer_b->draw_and_get_frame(empty_matrices);

    ASSERT_EQ(pixels_a.size(), pixels_b.size());
    ASSERT_NE(pixels_a, pixels_b);
}

// mirrors the vision thread pool: one renderer per thread, concurrent draws
// on the same device/queue
TEST(VkOffscreenTest, ConcurrentRenderers) {
    const auto backend = make_vulkan_backend();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int nb_threads = 8, nb_frames = 5, w = 16, h = 16;

    std::vector<std::thread> workers;
    std::vector<int> sums(nb_threads, -1);
    for (int t = 0; t < nb_threads; t++)
        workers.emplace_back([&, t] {
            const auto renderer =
                backend->make_offscreen_renderer(w, h, {0.f, 10.f, -2.f}, make_camera());
            renderer->make_current();
            renderer->add_drawable(
                "sky", backend->drawable_factory()->make_cube_map(file_reader, "cubemap/1"));

            const auto matrices = std::vector<std::tuple<std::string, glm::mat4>>{
                {"sky", glm::scale(glm::mat4(1.f), glm::vec3(2000.f))}};

            image<uint8_t> frame{};
            for (int i = 0; i < nb_frames; i++) frame = renderer->draw_and_get_frame(matrices);
            sums[t] = pixel_sum(frame.pixels);
        });
    for (auto &worker: workers) worker.join();

    for (const int sum: sums) ASSERT_GT(sum, 0);
}

TEST(VkOffscreenTest, ShadowRendererSmoke) {
    const auto backend = std::make_unique<VulkanBackend>();
    const auto file_reader =
        std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    constexpr int w = 32, h = 32;
    // internal ctor: with_shadows = true (the player-view configuration)
    const auto context = std::dynamic_pointer_cast<VulkanRenderContext>(backend->render_context());
    ASSERT_NE(context, nullptr);
    const auto renderer = std::make_unique<VulkanOffscreenRenderer>(
        context->device(), w, h, glm::vec3{0.f, 10.f, -2.f}, make_camera(), true);

    const std::vector<std::tuple<float, float, float>> vertices{
        {-5.f, -5.f, 0.f}, {0.f, 5.f, 0.f}, {5.f, -5.f, 0.f}};
    renderer->add_drawable(
        "tri", backend->drawable_factory()->make_diffuse(
                   file_reader, vertices, glm::vec4(1.f, 1.f, 1.f, 1.f)));

    const auto matrices = std::vector<std::tuple<std::string, glm::mat4>>{{"tri", glm::mat4(1.f)}};

    renderer->draw_and_get_frame(matrices);
    const auto [pixels] = renderer->draw_and_get_frame(matrices);
    ASSERT_GT(pixel_sum(pixels), 0);
}
