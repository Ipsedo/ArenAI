//
// Created by samuel on 29/09/2025.
//

#include <cstring>
#include <vector>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

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

    EGLConfig configs[64];
    EGLint num_config = 0;
    if (eglChooseConfig(display, config_attrib, configs, 64, &num_config) != EGL_TRUE
        || num_config < 1)
        throw std::runtime_error("Can't get EGLConfig");

    EGLConfig config = nullptr;
    for (EGLint i = 0; i < num_config; ++i) {
        EGLint r = 0, g = 0, b = 0, a = 0;
        eglGetConfigAttrib(display, configs[i], EGL_RED_SIZE, &r);
        eglGetConfigAttrib(display, configs[i], EGL_GREEN_SIZE, &g);
        eglGetConfigAttrib(display, configs[i], EGL_BLUE_SIZE, &b);
        eglGetConfigAttrib(display, configs[i], EGL_ALPHA_SIZE, &a);
        if (r == 8 && g == 8 && b == 8 && a == 8) {
            config = configs[i];
            break;
        }
    }
    if (config == nullptr) throw std::runtime_error("No 8-bit RGBA EGLConfig available");

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
    : Renderer(std::make_shared<PBufferGLContext>(main_context, width, height), light_pos, camera),
      width(width), height(height) {}

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

void PBufferRenderer::init_pbos() {
    pbo_size_ = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    glGenBuffers(NUM_PBOS, pbos_);
    for (int i = 0; i < NUM_PBOS; i++) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[i]);
        glBufferData(
            GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(pbo_size_), nullptr, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    pbo_initialized_ = true;
}

image<uint8_t> PBufferRenderer::draw_and_get_frame(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    draw(model_matrices);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    if (!pbo_initialized_) {
        init_pbos();

        // First frame: kick off async read into PBO 0, return a black frame
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[0]);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFlush();
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        pbo_index_ = 1;

        const int hw = width * height;
        return image(std::vector<uint8_t>(hw * 3, 0));
    }

    const int read_pbo = pbo_index_;
    const int map_pbo = 1 - pbo_index_;

    // Start async read of current frame into read_pbo
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[read_pbo]);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glFlush();

    // Map the previous frame's PBO (should be ready by now)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[map_pbo]);
    const auto *src = static_cast<const uint8_t *>(glMapBufferRange(
        GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(pbo_size_), GL_MAP_READ_BIT));

    const int hw = width * height;
    auto frame = image(std::vector<uint8_t>(hw * 3));

    if (src) {
        // RGBA HWC -> RGB CHW (drop alpha, separate channels)
        // glReadPixels returns rows bottom-to-top (OpenGL origin = bottom-left),
        // so flip vertically to get top-to-bottom image rows.
        auto *dst = frame.pixels.data();
        for (int y = 0; y < height; ++y) {
            const int src_row = (height - 1 - y) * width;
            const int dst_row = y * width;
            for (int x = 0; x < width; ++x) {
                const int s = (src_row + x) * 4;
                const int d = dst_row + x;
                dst[0 * hw + d] = src[s + 0];
                dst[1 * hw + d] = src[s + 1];
                dst[2 * hw + d] = src[s + 2];
            }
        }
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    pbo_index_ = 1 - pbo_index_;

    return frame;
}

int PBufferRenderer::get_width() const { return width; }

int PBufferRenderer::get_height() const { return height; }

PBufferRenderer::~PBufferRenderer() {
    if (pbo_initialized_) glDeleteBuffers(NUM_PBOS, pbos_);
}
