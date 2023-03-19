//
// Created by samuel on 18/03/2023.
//

#include "renderer.h"
#include "../utils/logging.h"

#include <string>
#include <utility>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <glm/gtc/matrix_transform.hpp>

Renderer::Renderer(ANativeWindow *window, std::shared_ptr<Camera> camera) :
        camera(std::move(camera)),
        drawables(),
        light_pos(0., 100., 0.),
        is_animating(true) {

    const EGLint attribs[] = {
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_NONE
    };
    EGLint w, h, format;
    EGLint numConfigs;
    EGLConfig config;
    EGLContext context;

    LOG_INFO("ici 1 %f", 1.f);

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    LOG_INFO("ici 2 %f", 1.f);

    eglInitialize(display, nullptr, nullptr);

    LOG_INFO("ici 3 %f", 2.f);

    /* Here, the application chooses the configuration it desires.
     * find the best match if possible, otherwise use the very first one
     */
    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    assert(supportedConfigs);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs, &numConfigs);
    assert(numConfigs);
    auto i = 0;
    for (; i < numConfigs; i++) {
        auto &cfg = supportedConfigs[i];
        EGLint r, g, b, d;
        if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r) &&
            eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
            eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b) &&
            eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
            r == 8 && g == 8 && b == 8 && d == 0) {

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
    context = eglCreateContext(display, config, nullptr, nullptr);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
        throw std::runtime_error("Unable to eglMakeCurrent");

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    width = w;
    height = h;

    // Check openGL on the system
    /*auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
    }*/

    glViewport(0, 0, width, height);

    glClearColor(0.5, 0., 0., 1.);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    //glEnable(GL_MULTISAMPLE);

    glDepthFunc(GL_LEQUAL);

    glDepthMask(GL_TRUE);

}

void Renderer::add_drawable(const std::string &name, const std::shared_ptr<Drawable> &drawable) {
    drawables.insert({name, drawable});
}

void Renderer::draw(std::map<std::string, glm::mat4> model_matrices) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view_matrix = glm::lookAt(
            camera->pos(),
            camera->look(),
            camera->up()
    );

    glm::mat4 proj_matrix = glm::frustum(
            -1.f, 1.f,
            -float(height) / float(width), float(height) / float(width),
            0.1f,
            2000.f * sqrt(3.f)
    );

    for (auto [name, drawable]: drawables) {
        glm::mat4 m_matrix = model_matrices[name];

        auto mv_matrix = view_matrix * m_matrix;
        auto mvp_matrix = proj_matrix * mv_matrix;

        drawable->draw(mvp_matrix, mv_matrix, light_pos, camera->pos());
    }

    eglSwapBuffers(display, surface);
}

void Renderer::enable() {
    is_animating = true;
}

void Renderer::disable() {
    is_animating = false;
}

bool Renderer::is_enabled() const {
    return is_animating;
}

void Renderer::close() {
    disable();

    drawables.clear();
}
