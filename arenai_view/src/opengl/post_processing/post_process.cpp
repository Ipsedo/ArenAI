//
// Created by samuel on 14/07/2026.
//

#include "./post_process.h"

#include <algorithm>
#include <stdexcept>

#include "./ao_blur.h"
#include "./bloom_blur.h"
#include "./bloom_bright.h"
#include "./composite.h"
#include "./god_rays.h"
#include "./ssao.h"

namespace arenai::view {

    std::vector<std::shared_ptr<AbstractPostProcessingEffect>>
    make_default_post_processing_effects(const int width, const int height) {
        return {
            std::make_shared<SsaoEffect>(width, height),
            std::make_shared<AoBlurEffect>(width, height),
            std::make_shared<BloomBrightEffect>(width, height),
            std::make_shared<BloomBlurEffect>(width, height),
            std::make_shared<GodRaysEffect>(width, height),
            std::make_shared<CompositeEffect>(width, height)};
    }

    PostProcess::PostProcess(
        const int width, const int height,
        std::vector<std::shared_ptr<AbstractPostProcessingEffect>> ordered_effects)
        : width(width), height(height), frame(0), msaa_fbo(0), msaa_color_rbo(0), msaa_depth_rbo(0),
          resolve_fbo(0), resolve_texture(0), resolve_depth_texture(0),
          ordered_post_processing_effects(std::move(ordered_effects)) {
        create_scene_targets();
    }

    void PostProcess::create_scene_targets() {
        GLint max_samples = 1;
        glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
        const auto samples = std::min(MSAA_SAMPLES, max_samples);

        glGenRenderbuffers(1, &msaa_color_rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, msaa_color_rbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, width, height);

        glGenRenderbuffers(1, &msaa_depth_rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, msaa_depth_rbo);
        glRenderbufferStorageMultisample(
            GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT24, width, height);

        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glGenFramebuffers(1, &msaa_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaa_color_rbo);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msaa_depth_rbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("MSAA framebuffer is incomplete");

        glGenTextures(1, &resolve_texture);
        glBindTexture(GL_TEXTURE_2D, resolve_texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // resolved depth, sampled by the SSAO and god-rays passes; the format
        // must match the MSAA depth renderbuffer for the depth resolve blit,
        // and depth textures require NEAREST filtering
        glGenTextures(1, &resolve_depth_texture);
        glBindTexture(GL_TEXTURE_2D, resolve_depth_texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &resolve_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resolve_texture, 0);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, resolve_depth_texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("resolve framebuffer is incomplete");

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void PostProcess::destroy_scene_targets() {
        glDeleteFramebuffers(1, &msaa_fbo);
        glDeleteFramebuffers(1, &resolve_fbo);
        glDeleteRenderbuffers(1, &msaa_color_rbo);
        glDeleteRenderbuffers(1, &msaa_depth_rbo);
        glDeleteTextures(1, &resolve_texture);
        glDeleteTextures(1, &resolve_depth_texture);

        msaa_fbo = msaa_color_rbo = msaa_depth_rbo = 0;
        resolve_fbo = resolve_texture = resolve_depth_texture = 0;
    }

    void PostProcess::resize(const int new_width, const int new_height) {
        if (new_width == width && new_height == height) return;

        width = new_width;
        height = new_height;

        destroy_scene_targets();
        create_scene_targets();

        for (const auto &effect: ordered_post_processing_effects) effect->resize(width, height);
    }

    void PostProcess::begin_scene_pass() const { glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo); }

    void PostProcess::draw_to_screen(const glm::mat4 &proj_matrix, const glm::vec3 &sun_dir_view) {
        frame = (frame + 1) % 1024;

        // resolve the MSAA samples into single-sampled color + depth textures
        // (NEAREST is mandatory as soon as the depth buffer is blitted)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo);
        glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
            GL_NEAREST);

        glDisable(GL_DEPTH_TEST);

        AbstractPostProcessingEffect::FrameContext context{
            resolve_texture,
            resolve_depth_texture,
            width,
            height,
            proj_matrix,
            // projection terms consumed by the depth-reconstruction shaders
            glm::vec4(proj_matrix[0][0], proj_matrix[1][1], proj_matrix[2][2], proj_matrix[3][2]),
            sun_dir_view,
            frame,
            {},
            {}};

        for (const auto &effect: ordered_post_processing_effects) effect->render(context);

        Program::disable_texture();

        glEnable(GL_DEPTH_TEST);
    }

    PostProcess::~PostProcess() { destroy_scene_targets(); }

}// namespace arenai::view
