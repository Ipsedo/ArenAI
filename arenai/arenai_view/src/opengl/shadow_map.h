//
// Created by samuel on 13/07/2026.
//

#ifndef ARENAI_SHADOW_MAP_H
#define ARENAI_SHADOW_MAP_H

#include <GLES3/gl3.h>

namespace arenai::view {

    // Depth-only framebuffer rendered from the light's point of view.
    // Must be created and used with the owning renderer's GL context current
    // (FBOs are not shared across EGL contexts).
    class ShadowMap {
    public:
        explicit ShadowMap(int size);

        ShadowMap(const ShadowMap &) = delete;
        ShadowMap &operator=(const ShadowMap &) = delete;

        void begin_depth_pass() const;
        void end_depth_pass() const;

        GLuint depth_texture() const;

        ~ShadowMap();

    private:
        int size;

        GLuint fbo_id;
        GLuint depth_texture_id;
    };

}// namespace arenai::view

#endif// ARENAI_SHADOW_MAP_H
