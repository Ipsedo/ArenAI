//
// Created by samuel on 29/09/2025.
//

#ifndef PHYVR_FRAMEBUFFER_RENDERER_H
#define PHYVR_FRAMEBUFFER_RENDERER_H

#include "./renderer.h"

#include <EGL/egl.h>
#include <array>
#include <vector>

typedef std::array<std::uint8_t, 4> pixel;

class FrameBufferContext : public AbstractGLContext {
public:
  FrameBufferContext(int width, int height);
  EGLDisplay get_display() override;

  EGLSurface get_surface() override;

  EGLContext get_context() override;

private:
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
};

class FrameBufferRenderer : public Renderer {
public:
  FrameBufferRenderer(const std::shared_ptr<Camera> &camera,
                      glm::vec3 light_pos);

  void add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable) override;

  std::vector<std::vector<pixel>> get_frame();

protected:
  void _on_end_frame() override;

private:
  std::vector<std::vector<pixel>> last_frame;
};

#endif // PHYVR_FRAMEBUFFER_RENDERER_H
