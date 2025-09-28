//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_ANDROID_RENDERER_H
#define PHYVR_ANDROID_RENDERER_H

#include <EGL/egl.h>
#include <android/native_activity.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <memory>
#include <phyvr_view/drawable.h>
#include <phyvr_view/renderer.h>

class NormalRenderer : public Renderer {
public:
  NormalRenderer(ANativeWindow *window, std::shared_ptr<Camera> camera);

  void add_drawable(const std::string &name,
                    std::unique_ptr<Drawable> drawable) override;

  void add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable) override;

  void remove_drawable(const std::string &name);

  int get_width() const override;
  int get_height() const override;

  void draw(const std::vector<std::tuple<std::string, glm::mat4>>
                &model_matrices) override;

  ~NormalRenderer();

private:
  int width;
  int height;

protected:
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;

  glm::vec3 light_pos;

  std::shared_ptr<Camera> camera;

  std::map<std::string, std::unique_ptr<Drawable>> drawables;
  std::vector<std::unique_ptr<HUDDrawable>> hud_drawables;
};
#endif // PHYVR_ANDROID_RENDERER_H
