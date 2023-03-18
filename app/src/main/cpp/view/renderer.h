//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <string>
#include <memory>
#include <map>
#include <android/native_window.h>
#include <EGL/egl.h>

#include "camera.h"
#include "drawable.h"

class Renderer {
public:
    Renderer(ANativeWindow *window, std::shared_ptr<Camera> camera);

    void add_drawable(const std::string &name, const std::shared_ptr<Drawable> &drawable);

    void draw(std::map<std::string, glm::mat4> model_matrices);

    void enable();

    void disable();

    bool is_enable() const;

    void close();

private:
    EGLDisplay display;
    int width;
    int height;

    glm::vec3 light_pos;
    std::shared_ptr<Camera> camera;

    std::map<std::string, std::shared_ptr<Drawable>> drawables;

    bool is_animating;
};

#endif //PHYVR_RENDERER_H
