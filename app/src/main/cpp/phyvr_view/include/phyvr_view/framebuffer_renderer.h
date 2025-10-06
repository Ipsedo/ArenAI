//
// Created by samuel on 29/09/2025.
//

#ifndef PHYVR_FRAMEBUFFER_RENDERER_H
#define PHYVR_FRAMEBUFFER_RENDERER_H

#include <array>
#include <vector>

#include <EGL/egl.h>
#include <glm/glm.hpp>

#include "./renderer.h"

template<typename T>
using image = std::array<std::vector<std::vector<T>>, 3>;

class PBufferGLContext : public AbstractGLContext {
public:
    PBufferGLContext();

    EGLDisplay get_display() override;

    EGLSurface get_surface() override;

    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
};

class PBufferRenderer : public Renderer {
public:
    PBufferRenderer(
        int width, int height, glm::vec3 light_pos, const std::shared_ptr<Camera> &camera);

    image<uint8_t>
    draw_and_get_frame(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices);

    ~PBufferRenderer() override;

protected:
    void on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;

    void on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;
};

#endif// PHYVR_FRAMEBUFFER_RENDERER_H
