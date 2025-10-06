//
// Created by samuel on 29/09/2025.
//
#include <array>
#include <cstring>
#include <iostream>
#include <vector>

#include <phyvr_view/framebuffer_renderer.h>

/*
 * Context
 */

PBufferGLContext::PBufferGLContext() : AbstractGLContext() {
    display = eglGetDisplay(EGL_NO_DISPLAY);

    const EGLint config_attrib[] = {
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        4,
        EGL_GREEN_SIZE,
        4,
        EGL_BLUE_SIZE,
        4,
        EGL_ALPHA_SIZE,
        0,
        EGL_DEPTH_SIZE,
        16,
        EGL_STENCIL_SIZE,
        8,
        EGL_SAMPLES,
        0,
        EGL_NONE};
    const EGLint context_attrib[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLint num_config = 0;
    EGLConfig config;

    eglChooseConfig(display, config_attrib, &config, 1, &num_config);

    context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrib);

    const EGLint pbattribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    surface = eglCreatePbufferSurface(display, config, pbattribs);
}

EGLDisplay PBufferGLContext::get_display() { return display; }

EGLSurface PBufferGLContext::get_surface() { return surface; }

EGLContext PBufferGLContext::get_context() { return context; }

/*
 * Frame Buffer
 */

PBufferRenderer::PBufferRenderer(
    int width, int height, glm::vec3 light_pos, const std::shared_ptr<Camera> &camera)
    : Renderer(std::make_shared<PBufferGLContext>(), width, height, light_pos, camera) {}

void PBufferRenderer::on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {

    glViewport(0, 0, get_width(), get_height());
    glScissor(0, 0, get_width(), get_height());

    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glEnable(GL_CULL_FACE);
    // glCullFace(GL_BACK);
    // glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PBufferRenderer::on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {}

std::vector<std::vector<pixel>> PBufferRenderer::draw_and_get_frame(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    draw(model_matrices);

    std::vector<std::vector<pixel>> image(get_height(), std::vector<pixel>(get_width()));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glFinish();

    const int width = get_width();
    const int height = get_height();

    std::vector<pixel> linear(static_cast<size_t>(width) * static_cast<size_t>(height));
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, linear.data());

    for (int y = 0; y < height; ++y) {
        const int src_y = y;
        const int dst_y = (height - 1 - y);
        std::memcpy(
            image[dst_y].data(), linear.data() + static_cast<size_t>(src_y) * width,
            sizeof(pixel) * static_cast<size_t>(width));
    }
    return image;
}

PBufferRenderer::~PBufferRenderer() = default;
