//
// Created by samuel on 18/03/2023.
//

#include "./android_gl_context.h"
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

AndroidGLContext::AndroidGLContext(ANativeWindow *window)
    : AbstractGLContext() {
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  eglInitialize(display, nullptr, nullptr);

  const EGLint config_attrib[] = {EGL_RENDERABLE_TYPE,
                                  EGL_OPENGL_ES3_BIT,
                                  EGL_SURFACE_TYPE,
                                  EGL_WINDOW_BIT,
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
  const EGLint context_attrib[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  EGLint width, height;
  EGLint num_config = 0;
  EGLConfig config;

  eglChooseConfig(display, config_attrib, &config, 1, &num_config);

  context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrib);

  surface = eglCreateWindowSurface(display, config, window, nullptr);

  eglMakeCurrent(display, surface, surface, context);

  eglQuerySurface(display, surface, EGL_WIDTH, &width);
  eglQuerySurface(display, surface, EGL_HEIGHT, &height);
}

EGLDisplay AndroidGLContext::get_display() { return display; }

EGLSurface AndroidGLContext::get_surface() { return surface; }

EGLContext AndroidGLContext::get_context() { return context; }
