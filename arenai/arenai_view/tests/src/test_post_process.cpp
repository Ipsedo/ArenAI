//
// Created by samuel on 15/07/2026.
//

#include <cmath>
#include <numeric>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include "opengl/post_processing/post_process.h"
#include "opengl/renderers/gl_offscreen_renderer.h"

using namespace arenai;
using namespace arenai::view;

namespace {

    constexpr int WIDTH = 64, HEIGHT = 64;

    // smoke-test of the whole player post-processing pipeline on a headless
    // pbuffer context: every effect shader (SSAO, blurs, bloom, god rays,
    // composite) is compiled, linked and drawn once
    void run_frame(PostProcess &post_process, const int width, const int height) {
        const glm::mat4 proj_matrix = glm::perspective(
            glm::quarter_pi<float>(), static_cast<float>(width) / static_cast<float>(height), 1.f,
            2000.f * std::sqrt(3.f));
        // sun in front of the camera, so the god-rays pass runs too
        const glm::vec3 sun_dir_view = glm::normalize(glm::vec3(0.1f, 0.3f, -1.f));

        post_process.begin_scene_pass();
        glViewport(0, 0, width, height);
        glClearColor(0.5f, 0.6f, 0.7f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        post_process.draw_to_screen(proj_matrix, sun_dir_view);
    }

    long screen_pixel_sum(const int width, const int height) {
        std::vector<uint8_t> pixels(4 * width * height);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        return std::accumulate(
            pixels.begin(), pixels.end(), 0L,
            [](const long acc, const uint8_t p) { return acc + static_cast<long>(p); });
    }

}// namespace

TEST(TestPostProcess, FullPipelineRuns) {
    const auto main_context = std::make_shared<HeadlessEglContext>();
    const auto context = std::make_shared<PBufferContext>(main_context, WIDTH, HEIGHT);
    context->make_current();

    PostProcess post_process(WIDTH, HEIGHT, make_default_post_processing_effects(WIDTH, HEIGHT));

    run_frame(post_process, WIDTH, HEIGHT);

    ASSERT_EQ(glGetError(), GL_NO_ERROR);
    // the cleared scene must reach the screen through the composite pass
    ASSERT_GT(screen_pixel_sum(WIDTH, HEIGHT), 0L);

    context->release_current();
}

TEST(TestPostProcess, SurvivesResize) {
    const auto main_context = std::make_shared<HeadlessEglContext>();
    const auto context = std::make_shared<PBufferContext>(main_context, WIDTH, HEIGHT);
    context->make_current();

    PostProcess post_process(WIDTH, HEIGHT, make_default_post_processing_effects(WIDTH, HEIGHT));
    run_frame(post_process, WIDTH, HEIGHT);

    // shrinking only recreates the render targets: the pbuffer surface stays
    // large enough for the read-back
    constexpr int NEW_WIDTH = 32, NEW_HEIGHT = 48;
    post_process.resize(NEW_WIDTH, NEW_HEIGHT);
    run_frame(post_process, NEW_WIDTH, NEW_HEIGHT);

    ASSERT_EQ(glGetError(), GL_NO_ERROR);
    ASSERT_GT(screen_pixel_sum(NEW_WIDTH, NEW_HEIGHT), 0L);

    context->release_current();
}
