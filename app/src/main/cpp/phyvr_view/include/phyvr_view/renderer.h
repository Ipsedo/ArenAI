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

#include <EGL/egl.h>
#include <glm/glm.hpp>

#include <phyvr_view/camera.h>
#include <phyvr_view/drawable.h>
#include <phyvr_view/hud.h>

#include "./camera.h"
#include "./drawable.h"
#include "./hud.h"

class AbstractGLContext {
public:
  AbstractGLContext();
  virtual EGLDisplay get_display() = 0;
  virtual EGLSurface get_surface() = 0;
  virtual EGLContext get_context() = 0;

  void make_current();

private:
  bool current_called;
};

class Renderer {
public:
  Renderer(
    const std::shared_ptr<AbstractGLContext> &gl_context, int width, int height,
    glm::vec3 light_pos, const std::shared_ptr<Camera> &camera);
  virtual void add_drawable(const std::string &name, std::unique_ptr<Drawable> drawable);
  virtual void remove_drawable(const std::string &name);

  void draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices);

  virtual int get_width() const;
  virtual int get_height() const;

  virtual ~Renderer();

protected:
  virtual void on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) = 0;

  virtual void on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) = 0;

private:
  int width;
  int height;

  glm::vec3 light_pos;

  std::map<std::string, std::unique_ptr<Drawable>> drawables;

  std::shared_ptr<AbstractGLContext> gl_context;

  std::shared_ptr<Camera> camera;
};

class PlayerRenderer : public Renderer {
public:
  PlayerRenderer(
    const std::shared_ptr<AbstractGLContext> &gl_context, int width, int height,
    const glm::vec3 &lightPos, const std::shared_ptr<Camera> &camera);

  virtual void add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable);

  ~PlayerRenderer() override;

protected:
  void on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;

  void on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) override;

private:
  std::vector<std::unique_ptr<HUDDrawable>> hud_drawables;
};

#endif// PHYVR_RENDERER_H
