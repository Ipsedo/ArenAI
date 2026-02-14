//
// Created by samuel on 29/09/2025.
//

#include <cstring>
#include <vector>

#include <arenai_view/pbuffer_renderer.h>

#include "arenai_view/errors.h"

/*
 * Context
 */

PBufferGLContext::PBufferGLContext(
    const std::shared_ptr<AbstractGLContext> &main_context, int width, int height)
    : AbstractGLContext(), display(main_context->get_display()) {
    const EGLint config_attrib[] = {
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_DEPTH_SIZE,
        16,
        EGL_STENCIL_SIZE,
        8,
        EGL_SAMPLES,
        0,
        EGL_NONE};
    constexpr EGLint context_attrib[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLint num_config = 0;
    EGLConfig config;

    if (eglChooseConfig(display, config_attrib, &config, 1, &num_config) != EGL_TRUE)
        throw std::runtime_error("Can't get EGLConfig");

    context = eglCreateContext(display, config, main_context->get_context(), context_attrib);
    if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext failed");

    const EGLint pbattribs[] = {EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE};
    surface = eglCreatePbufferSurface(display, config, pbattribs);
    if (surface == EGL_NO_SURFACE) throw std::runtime_error("eglCreatePbufferSurface failed");

    eglMakeCurrent(display, surface, surface, context);

    eglQuerySurface(display, surface, EGL_WIDTH, &width);
    eglQuerySurface(display, surface, EGL_HEIGHT, &height);

    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

EGLDisplay PBufferGLContext::get_display() { return display; }

EGLSurface PBufferGLContext::get_surface() { return surface; }

EGLContext PBufferGLContext::get_context() { return context; }

/*
 * Frame Buffer
 */

PBufferRenderer::PBufferRenderer(
    const std::shared_ptr<AbstractGLContext> &main_context, const int width, const int height,
    const glm::vec3 light_pos, const std::shared_ptr<Camera> &camera)
    : Renderer(
        std::make_shared<PBufferGLContext>(main_context, width, height), width, height, light_pos,
        camera) {}

void PBufferRenderer::on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {
    glViewport(0, 0, get_width(), get_height());

    glClearColor(1., 0., 0., 0.);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PBufferRenderer::on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {}

std::shared_ptr<image<uint8_t>> PBufferRenderer::draw_and_get_frame(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    draw(model_matrices);

    const int width = get_width();
    const int height = get_height();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    constexpr int in_channels = 4;
    std::vector<unsigned char> linear(
        static_cast<size_t>(width) * static_cast<size_t>(height) * in_channels);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, linear.data());

    const int hw = width * height;

    auto frame = std::make_shared<image<uint8_t>>(std::vector<uint8_t>(hw * 3));

    for (int y = 0; y < height; ++y) {
        const int src_y = y;
        const int dst_y = height - 1 - y;

        const uint8_t *src = linear.data() + src_y * width * in_channels;

        for (int x = 0; x < width; ++x) {
            const int dst = dst_y * width + x;

            frame->pixels[0 * hw + dst] = src[in_channels * x + 0];
            frame->pixels[1 * hw + dst] = src[in_channels * x + 1];
            frame->pixels[2 * hw + dst] = src[in_channels * x + 2];
        }
    }

    return frame;
}

PBufferRenderer::~PBufferRenderer() = default;
