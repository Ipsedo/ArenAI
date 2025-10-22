//
// Created by samuel on 29/09/2025.
//

#ifndef ARENAI_PBUFFER_RENDERER_H
#define ARENAI_PBUFFER_RENDERER_H

#include <array>
#include <vector>

#include <EGL/egl.h>
#include <glm/glm.hpp>

#include "./renderer.h"

// Indexing :
// Channels x Height x Width
template<typename T>
using image = std::vector<std::vector<std::vector<T>>>;

class PBufferGLContext final : public AbstractGLContext {
public:
    explicit PBufferGLContext(
        const std::shared_ptr<AbstractGLContext> &main_context, int width, int height);

    EGLDisplay get_display() override;

    EGLSurface get_surface() override;

    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
};

class PBufferRenderer final : public Renderer {
public:
    PBufferRenderer(
        const std::shared_ptr<AbstractGLContext> &main_context, int width, int height,
        glm::vec3 light_pos, const std::shared_ptr<Camera> &camera);

    image<uint8_t>
    draw_and_get_frame(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices);

    ~PBufferRenderer() override;

protected:
    void on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;

    void on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;
};

#endif// ARENAI_PBUFFER_RENDERER_H
