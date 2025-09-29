//
// Created by samuel on 29/09/2025.
//

#include <glm/gtc/matrix_transform.hpp>
#include <phyvr_view/errors.h>
#include <phyvr_view/renderer.h>

Renderer::Renderer(const std::shared_ptr<AbstractGLContext> &gl_context,
                   int width, int height, glm::vec3 light_pos,
                   const std::shared_ptr<Camera> &camera)
    : display(gl_context->get_display()), surface(gl_context->get_surface()),
      context(gl_context->get_context()), width(width), height(height),
      light_pos(light_pos), camera(camera) {}

void Renderer::add_drawable(const std::string &name,
                            std::unique_ptr<Drawable> drawable) {
  drawables.insert({name, std::move(drawable)});
}

void Renderer::add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable) {
  hud_drawables.push_back(std::move(hud_drawable));
}

void Renderer::remove_drawable(const std::string &name) {
  drawables.erase(name);
}

void Renderer::draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {

  // set-up
  glViewport(0, 0, width, height);

  glClearColor(1., 1., 1., 0.);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_TRUE);

  glDisable(GL_BLEND);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw
  glm::mat4 view_matrix =
      glm::lookAt(camera->pos(), camera->look(), camera->up());

  glm::mat4 proj_matrix =
      glm::perspective(float(M_PI) / 4.f, float(width) / float(height), 1.f,
                       2000.f * std::sqrtf(3.f));

  glEnable(GL_DEPTH_TEST);

  for (const auto &[name, m_matrix] : model_matrices) {
    auto mv_matrix = view_matrix * m_matrix;
    auto mvp_matrix = proj_matrix * mv_matrix;

    drawables[name]->draw(mvp_matrix, mv_matrix, light_pos, camera->pos());
  }

  glDisable(GL_DEPTH_TEST);

  for (auto &hud_drawable : hud_drawables)
    hud_drawable->draw(width, height);

  _on_end_frame();

  check_gl_error("draw");
}

int Renderer::get_width() const { return width; }

int Renderer::get_height() const { return height; }

EGLDisplay Renderer::_get_display() { return display; }

EGLSurface Renderer::_get_surface() { return surface; }

EGLContext Renderer::_get_context() { return context; }

Renderer::~Renderer() {
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
