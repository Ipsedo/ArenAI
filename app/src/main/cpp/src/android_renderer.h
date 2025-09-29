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

class AndroidGLContext : public AbstractGLContext {
public:
  explicit AndroidGLContext(ANativeWindow *window);
  EGLDisplay get_display() override;
  EGLSurface get_surface() override;
  EGLContext get_context() override;

private:
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
};

class NormalRenderer : public Renderer {
public:
  NormalRenderer(ANativeWindow *window, const std::shared_ptr<Camera> &camera);

protected:
  void _on_end_frame() override;
};
#endif // PHYVR_ANDROID_RENDERER_H
