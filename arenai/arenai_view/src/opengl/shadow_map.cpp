//
// Created by samuel on 13/07/2026.
//

#include "./shadow_map.h"

#include <stdexcept>

namespace arenai::view {

    ShadowMap::ShadowMap(const int size) : size(size), fbo_id(0), depth_texture_id(0) {
        glGenTextures(1, &depth_texture_id);
        glBindTexture(GL_TEXTURE_2D, depth_texture_id);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, size, size);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // sampler2DShadow comparison (hardware PCF)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &fbo_id);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_id, 0);

        constexpr GLenum no_color_buffer = GL_NONE;
        glDrawBuffers(1, &no_color_buffer);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("shadow map framebuffer is incomplete");

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void ShadowMap::begin_depth_pass() const {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
        glViewport(0, 0, size, size);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);

        glClear(GL_DEPTH_BUFFER_BIT);

        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.f, 4.f);
    }

    void ShadowMap::end_depth_pass() const {
        glDisable(GL_POLYGON_OFFSET_FILL);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    GLuint ShadowMap::depth_texture() const { return depth_texture_id; }

    ShadowMap::~ShadowMap() {
        glDeleteFramebuffers(1, &fbo_id);
        glDeleteTextures(1, &depth_texture_id);
    }

}// namespace arenai::view
