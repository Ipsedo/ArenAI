//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "./camera.h"
#include "./drawable.h"
#include "./hud.h"
#include <EGL/egl.h>
#include <glm/glm.hpp>
#include <phyvr_view/camera.h>
#include <phyvr_view/drawable.h>
#include <phyvr_view/hud.h>

class AbstractGLContext {
public:
  virtual EGLDisplay get_display() = 0;
  virtual EGLSurface get_surface() = 0;
  virtual EGLContext get_context() = 0;
};

class Renderer {
public:
  Renderer(const std::shared_ptr<AbstractGLContext> &gl_context, int width,
           int height, glm::vec3 light_pos,
           const std::shared_ptr<Camera> &camera);
  virtual void add_drawable(const std::string &name,
                            std::unique_ptr<Drawable> drawable);
  virtual void remove_drawable(const std::string &name);

  virtual void add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable);
  void
  draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices);

  virtual int get_width() const;
  virtual int get_height() const;

  virtual ~Renderer();

protected:
  virtual void _on_end_frame() = 0;

  EGLDisplay _get_display();
  EGLSurface _get_surface();
  EGLContext _get_context();

private:
  int width;
  int height;

  glm::vec3 light_pos;

  std::shared_ptr<Camera> camera;

  std::map<std::string, std::unique_ptr<Drawable>> drawables;
  std::vector<std::unique_ptr<HUDDrawable>> hud_drawables;

  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
};

#endif // PHYVR_RENDERER_H
