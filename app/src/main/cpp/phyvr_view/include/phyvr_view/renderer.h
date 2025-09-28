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

#include <phyvr_view/camera.h>
#include <phyvr_view/drawable.h>
#include <phyvr_view/hud.h>

class Renderer {
public:
  virtual void add_drawable(const std::string &name,
                            std::unique_ptr<Drawable> drawable) = 0;

  virtual void add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable) = 0;
  virtual void draw(const std::vector<std::tuple<std::string, glm::mat4>>
                        &model_matrices) = 0;

  virtual int get_width() const = 0;
  virtual int get_height() const = 0;

  virtual ~Renderer() = default;
};

#endif // PHYVR_RENDERER_H
