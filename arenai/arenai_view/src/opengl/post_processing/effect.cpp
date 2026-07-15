//
// Created by samuel on 15/07/2026.
//

#include "./effect.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "../constants.h"

namespace arenai::view {

    // fullscreen triangle covering the whole clip space, UVs derived in the
    // vertex shader
    const std::vector<float> FULLSCREEN_TRIANGLE = {-1.f, -1.f, 3.f, -1.f, -1.f, 3.f};

    // shaders are embedded, so no file reader is needed (only textures loaded
    // from disk would use it)
    Program::Builder AbstractPostProcessingEffect::effect_builder(const char *fragment_shader) {
        return Program::Builder(
                   nullptr, std::filesystem::path("post_vs.glsl"),
                   std::filesystem::path(fragment_shader))
            .add_buffer("vertices_buffer", FULLSCREEN_TRIANGLE)
            .add_attribute("a_position");
    }

    AbstractPostProcessingEffect::AbstractPostProcessingEffect(
        std::unique_ptr<Program> program, std::vector<TargetSpec> specs, const int width,
        const int height)
        : program(std::move(program)), specs(std::move(specs)) {
        create_targets(width, height);
    }

    AbstractPostProcessingEffect::~AbstractPostProcessingEffect() { destroy_targets(); }

    AbstractPostProcessingEffect::Target AbstractPostProcessingEffect::create_target(
        const GLenum internal_format, const int width, const int height) {
        Target target;
        target.width = width;
        target.height = height;

        glGenTextures(1, &target.texture);
        glBindTexture(GL_TEXTURE_2D, target.texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &target.fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("effect framebuffer is incomplete");

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        return target;
    }

    void AbstractPostProcessingEffect::create_targets(const int width, const int height) {
        for (const auto &[internal_format, size_divisor]: specs)
            targets.push_back(create_target(
                internal_format, std::max(1, width / size_divisor),
                std::max(1, height / size_divisor)));
    }

    void AbstractPostProcessingEffect::destroy_targets() {
        for (auto &target: targets) {
            glDeleteFramebuffers(1, &target.fbo);
            glDeleteTextures(1, &target.texture);
        }
        targets.clear();
    }

    void AbstractPostProcessingEffect::resize(const int new_width, const int new_height) {
        destroy_targets();
        create_targets(new_width, new_height);
    }

    void AbstractPostProcessingEffect::draw_fullscreen(
        const GLuint fbo, const int viewport_width, const int viewport_height) const {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, viewport_width, viewport_height);

        program->attrib("a_position", "vertices_buffer", 2, 2 * BYTES_PER_FLOAT, 0);

        Program::draw_arrays(GL_TRIANGLES, 0, 3);

        program->disable_attrib_array();
    }

}// namespace arenai::view
