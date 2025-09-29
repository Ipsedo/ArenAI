//
// Created by samuel on 29/09/2025.
//
#include <array>
#include <glm/glm.hpp>
#include <iostream>
#include <phyvr_view/framebuffer_renderer.h>
#include <vector>

FrameBufferRenderer::FrameBufferRenderer(const std::shared_ptr<Camera> &camera,
                                         glm::vec3 light_pos)
    : Renderer(std::make_shared<FrameBufferContext>(256, 256), 256, 256,
               light_pos, camera) {}

void FrameBufferRenderer::add_hud_drawable(
    std::unique_ptr<HUDDrawable> hud_drawable) {
  throw std::runtime_error("can't add hud drawable to RL agent renderer");
}

void FrameBufferRenderer::_on_end_frame() {
  std::vector<std::uint8_t> rgba(get_width() * get_height() * 4);

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, get_width(), get_height(), GL_RGBA, GL_UNSIGNED_BYTE,
               rgba.data());

  std::vector<std::vector<pixel>> image(get_height(),
                                        std::vector<pixel>(get_width()));
  for (int y = 0; y < get_height(); ++y) {
    for (int x = 0; x < get_width(); ++x) {
      size_t idx = (static_cast<size_t>(y) * get_width() + x) * 4;
      image[y][x] = {rgba[idx + 0], rgba[idx + 1], rgba[idx + 2],
                     rgba[idx + 3]};
    }
  }

  last_frame = image;
}

std::vector<std::vector<pixel>> FrameBufferRenderer::get_frame() {
  return last_frame;
}

/*
 * GL Context
 */

FrameBufferContext::FrameBufferContext(int width, int height) {
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display == EGL_NO_DISPLAY)
    throw std::runtime_error("Can't create display");

  EGLint major = 0, minor = 0;
  if (eglInitialize(display, &major, &minor) != EGL_TRUE)
    throw std::runtime_error("Can't initialise EGL");

  const EGLint config_attributes[] = {EGL_SURFACE_TYPE,
                                      EGL_PBUFFER_BIT,
                                      EGL_RENDERABLE_TYPE,
                                      EGL_OPENGL_ES3_BIT,
                                      EGL_RED_SIZE,
                                      8,
                                      EGL_GREEN_SIZE,
                                      8,
                                      EGL_BLUE_SIZE,
                                      8,
                                      EGL_ALPHA_SIZE,
                                      8,
                                      EGL_DEPTH_SIZE,
                                      24,
                                      EGL_NONE};
  EGLConfig config;
  EGLint num_configs = 0;
  if (eglChooseConfig(display, config_attributes, &config, 1, &num_configs) !=
          EGL_TRUE ||
      num_configs == 0)
    throw std::runtime_error("Can't choose config");

  const EGLint pbuf_attributes[] = {EGL_WIDTH, width, EGL_HEIGHT, height,
                                    EGL_NONE};
  surface = eglCreatePbufferSurface(display, config, pbuf_attributes);

  if (surface == EGL_NO_SURFACE)
    throw std::runtime_error("Can't create surface");

  if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE)
    throw std::runtime_error("Can't bind API");

  const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);

  if (context == EGL_NO_CONTEXT)
    throw std::runtime_error("Can't create context");

  if (eglMakeCurrent(display, surface, surface, context) != EGL_TRUE)
    throw std::runtime_error("Can't make context");
}

EGLDisplay FrameBufferContext::get_display() { return display; }

EGLSurface FrameBufferContext::get_surface() { return surface; }

EGLContext FrameBufferContext::get_context() { return context; }
