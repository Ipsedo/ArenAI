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
                               std::shared_ptr<Camera> camera)
    : camera(std::move(camera)), drawables(), hud_drawables(),
      light_pos(500., 2000., 1000.), width(0), height(0) {

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

  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  eglInitialize(display, nullptr, nullptr);

  /* Here, the application chooses the configuration it desires.
   * find the best match if possible, otherwise use the very first one
   */
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

  /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
   * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
   * As soon as we picked a EGLConfig, we can safely reconfigure the
   * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
  eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
  surface = eglCreateWindowSurface(display, config, window, nullptr);
  context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrib);

  if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    throw std::runtime_error("Unable to eglMakeCurrent");

  eglQuerySurface(display, surface, EGL_WIDTH, &w);
  eglQuerySurface(display, surface, EGL_HEIGHT, &h);

  width = w;
  height = h;

  // Check openGL on the system
  auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
  for (auto name : opengl_info) {
    auto info = glGetString(name);
    LOG_DEBUG("GL_INFO \"%d\" %s", name, info);
  }

  glViewport(0, 0, width, height);

  glClearColor(1., 1., 1., 0.);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_TRUE);

  glDisable(GL_BLEND);
}

void NormalRenderer::add_drawable(const std::string &name,
                                  std::unique_ptr<Drawable> drawable) {
  drawables.insert({name, std::move(drawable)});
}

void NormalRenderer::add_hud_drawable(
    std::unique_ptr<HUDDrawable> hud_drawable) {
  hud_drawables.push_back(std::move(hud_drawable));
}

void NormalRenderer::remove_drawable(const std::string &name) {
  drawables.erase(name);
}

void NormalRenderer::draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glm::mat4 view_matrix =
      glm::lookAt(camera->pos(), camera->look(), camera->up());

  glm::mat4 proj_matrix = glm::perspective(
      float(M_PI) / 4.f, float(width) / float(height), 1.f, 2000.f * sqrt(3.f));

  glEnable(GL_DEPTH_TEST);

  for (auto [name, m_matrix] : model_matrices) {
    auto mv_matrix = view_matrix * m_matrix;
    auto mvp_matrix = proj_matrix * mv_matrix;

    drawables[name]->draw(mvp_matrix, mv_matrix, light_pos, camera->pos());
  }

  glDisable(GL_DEPTH_TEST);

  for (auto &hud_drawable : hud_drawables)
    hud_drawable->draw(width, height);

  eglSwapBuffers(display, surface);

  check_gl_error("draw");
}

int NormalRenderer::get_width() const { return width; }

int NormalRenderer::get_height() const { return height; }

NormalRenderer::~NormalRenderer() {
  drawables.clear();

  if (display != EGL_NO_DISPLAY) {
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    if (context != EGL_NO_CONTEXT)
      eglDestroyContext(display, context);

    if (surface != EGL_NO_SURFACE)
      eglDestroySurface(display, surface);

    eglTerminate(display);
  }

  display = EGL_NO_DISPLAY;
  context = EGL_NO_CONTEXT;
  surface = EGL_NO_SURFACE;
}
