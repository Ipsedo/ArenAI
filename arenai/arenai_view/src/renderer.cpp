//
// Created by samuel on 29/09/2025.
//

#include <glm/gtc/matrix_transform.hpp>

#include <arenai_utils/logging.h>
#include <arenai_view/errors.h>
#include <arenai_view/renderer.h>

AbstractGLContext::AbstractGLContext() {}

void AbstractGLContext::make_current() {

    if (eglMakeCurrent(get_display(), get_surface(), get_surface(), get_context()) != EGL_TRUE)
        throw std::runtime_error("Can't make context");
}

void AbstractGLContext::release_current() {
    eglMakeCurrent(get_display(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

/*
 * Renderer
 */

Renderer::Renderer(
    const std::shared_ptr<AbstractGLContext> &gl_context, const int width, const int height,
    const glm::vec3 light_pos, const std::shared_ptr<Camera> &camera)
    : width(width), height(height), light_pos(light_pos), gl_context(gl_context), camera(camera) {}

void Renderer::add_drawable(const std::string &name, std::unique_ptr<Drawable> drawable) {
    drawables.insert({name, std::move(drawable)});
}

void Renderer::remove_drawable(const std::string &name) { drawables.erase(name); }

void Renderer::draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    on_new_frame(gl_context);

    // on_draw
    const glm::mat4 view_matrix = glm::lookAt(camera->pos(), camera->look(), camera->up());

    const glm::mat4 proj_matrix = glm::perspective(
        static_cast<float>(M_PI) / 4.f, static_cast<float>(width) / static_cast<float>(height), 1.f,
        2000.f * std::sqrtf(3.f));

    for (const auto &[name, m_matrix]: model_matrices) {
        auto mv_matrix = view_matrix * m_matrix;
        const auto mvp_matrix = proj_matrix * mv_matrix;

        drawables[name]->draw(mvp_matrix, mv_matrix, light_pos, camera->pos());
    }

    on_end_frame(gl_context);
}

int Renderer::get_width() const { return width; }

int Renderer::get_height() const { return height; }

void Renderer::make_current() const { gl_context->make_current(); }

void Renderer::release_current() const { gl_context->release_current(); }

Renderer::~Renderer() {
    drawables.clear();

    const auto display = gl_context->get_display();
    const auto surface = gl_context->get_surface();
    const auto context = gl_context->get_context();

    if (display != EGL_NO_DISPLAY) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);

        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
    }
}

/*
 * UI Renderer
 */

PlayerRenderer::PlayerRenderer(
    const std::shared_ptr<AbstractGLContext> &gl_context, const int width, const int height,
    const glm::vec3 &lightPos, const std::shared_ptr<Camera> &camera)
    : Renderer(gl_context, width, height, lightPos, camera), hud_drawables() {}

void PlayerRenderer::add_hud_drawable(std::unique_ptr<HUDDrawable> hud_drawable) {
    hud_drawables.push_back(std::move(hud_drawable));
}

void PlayerRenderer::on_end_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {

    glDisable(GL_DEPTH_TEST);

    for (const auto &hud_drawable: hud_drawables) hud_drawable->draw(get_width(), get_height());
    eglSwapBuffers(gl_context->get_display(), gl_context->get_surface());

    glEnable(GL_DEPTH_TEST);
}

void PlayerRenderer::on_new_frame(const std::shared_ptr<AbstractGLContext> &gl_context) {
    glViewport(0, 0, get_width(), get_height());

    glClearColor(1., 0., 0., 0.);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

PlayerRenderer::~PlayerRenderer() { hud_drawables.clear(); }
