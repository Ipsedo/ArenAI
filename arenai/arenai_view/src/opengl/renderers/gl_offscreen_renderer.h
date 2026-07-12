//
// Created by samuel on 29/09/2025.
//

#ifndef ARENAI_GL_OFFSCREEN_RENDERER_H
#define ARENAI_GL_OFFSCREEN_RENDERER_H

#include <memory>

#include <GLES3/gl3.h>

#include "../egl_render_context.h"
#include "./gl_renderer.h"

namespace arenai::view {

    class PBufferContext final : public EglRenderContext {
    public:
        PBufferContext(
            const std::shared_ptr<EglRenderContext> &main_context, int width, int height);

        EGLDisplay get_display() override;
        EGLSurface get_surface() override;
        EGLContext get_context() override;

        ~PBufferContext() override;

    private:
        EGLDisplay display;
        EGLSurface surface;
        EGLContext context;
    };

    class GlOffscreenRenderer final : public GlRenderer, public AbstractOffscreenRenderer {
    public:
        GlOffscreenRenderer(
            const std::shared_ptr<EglRenderContext> &main_context, int width, int height,
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
            bool with_shadows = false);

        image<uint8_t> draw_and_get_frame(
            const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        int get_width() const override;
        int get_height() const override;

        ~GlOffscreenRenderer() override;

    protected:
        void on_new_frame() override;
        void on_end_frame() override;

    private:
        int width, height;

        static constexpr int NUM_PBOS = 2;
        GLuint pbos_[NUM_PBOS]{};
        int pbo_index_{0};
        bool pbo_initialized_{false};
        size_t pbo_size_{0};

        void init_pbos();
    };

}// namespace arenai::view

#endif// ARENAI_GL_OFFSCREEN_RENDERER_H
