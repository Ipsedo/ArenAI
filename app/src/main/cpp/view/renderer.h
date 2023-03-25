//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_RENDERER_H
#define PHYVR_RENDERER_H

#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <vector>
#include <android/native_window.h>
#include <EGL/egl.h>

#include "./camera.h"
#include "./drawable/drawable.h"

class Renderer {
public:
    Renderer(ANativeWindow *window, std::shared_ptr<Camera> camera);

    void add_drawable(const std::string &name, std::unique_ptr<Drawable> drawable);

    void remove_drawable(const std::string &name);

    void draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices);

    ~Renderer();

private:
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;

    int width;
    int height;

    glm::vec3 light_pos;
    std::shared_ptr<Camera> camera;

    std::map<std::string, std::unique_ptr<Drawable>> drawables;
};

#endif //PHYVR_RENDERER_H
