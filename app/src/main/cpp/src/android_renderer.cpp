//
// Created by samuel on 18/03/2023.
//

#include "./android_renderer.h"
#include <phyvr_utils/logging.h>
#include <phyvr_view/errors.h>

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <utility>

/*
 * Normal Renderer
 */

NormalRenderer::NormalRenderer(ANativeWindow *window,
                               const std::shared_ptr<Camera> &camera)
    : Renderer(std::make_shared<AndroidGLContext>(window),
               ANativeWindow_getWidth(window), ANativeWindow_getHeight(window),
               glm::vec3(), camera) {}

void NormalRenderer::_on_end_frame() {
  eglSwapBuffers(_get_display(), _get_surface());
}

/*
 * Context
 */

EGLDisplay AndroidGLContext::get_display() { return display; }

AndroidGLContext::AndroidGLContext(ANativeWindow *window) {
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  eglInitialize(display, nullptr, nullptr);

  const EGLint config_attrib[] = {EGL_SURFACE_TYPE,
                                  EGL_WINDOW_BIT,
                                  EGL_BLUE_SIZE,
                                  8,
                                  EGL_GREEN_SIZE,
                                  8,
                                  EGL_RED_SIZE,
                                  8,
                                  EGL_ALPHA_SIZE,
                                  8,
                                  EGL_DEPTH_SIZE,
                                  16,
                                  EGL_STENCIL_SIZE,
                                  0,
                                  EGL_RENDERABLE_TYPE,
                                  EGL_OPENGL_ES3_BIT,
                                  EGL_NONE};
  const EGLint context_attrib[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  EGLint w, h, format;
  EGLint numConfigs;
  EGLConfig config;

  eglChooseConfig(display, config_attrib, nullptr, 0, &numConfigs);
  std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
  assert(supportedConfigs);
  eglChooseConfig(display, config_attrib, supportedConfigs.get(), numConfigs,
                  &numConfigs);
  assert(numConfigs);
  auto i = 0;
  for (; i < numConfigs; i++) {
    auto &cfg = supportedConfigs[i];
    EGLint r, g, b, d, s;
    if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r) &&
        eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
        eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b) &&
        eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
        eglGetConfigAttrib(display, cfg, EGL_STENCIL_SIZE, &s) && r == 8 &&
        g == 8 && b == 8 && d == 16 && s == 0) {

      config = supportedConfigs[i];
      break;
    }
  }
  if (i == numConfigs) {
    config = supportedConfigs[0];
  }

  eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
  surface = eglCreateWindowSurface(display, config, window, nullptr);

  context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrib);

  if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    throw std::runtime_error("Unable to eglMakeCurrent");

  eglQuerySurface(display, surface, EGL_WIDTH, &w);
  eglQuerySurface(display, surface, EGL_HEIGHT, &h);
}

EGLSurface AndroidGLContext::get_surface() { return surface; }

EGLContext AndroidGLContext::get_context() { return context; }
