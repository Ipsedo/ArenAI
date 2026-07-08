//
// Created by samuel on 29/09/2025.
//

#include "./gl_renderer.h"

#include <cmath>
#include <utility>

#include <GLES3/gl3.h>
#include <glm/gtc/matrix_transform.hpp>

using namespace arenai;

namespace arenai::view {

    /*
     * GlRenderer
     */

    GlRenderer::GlRenderer(
        std::shared_ptr<EglRenderContext> gl_context, const glm::vec3 light_pos,
        std::shared_ptr<AbstractCamera> camera)
        : light_pos(light_pos), gl_context(std::move(gl_context)), camera(std::move(camera)) {}

    void
    GlRenderer::add_drawable(const std::string &name, std::unique_ptr<AbstractDrawable> drawable) {
        drawables.insert({name, std::move(drawable)});
    }

    void GlRenderer::remove_drawable(const std::string &name) { drawables.erase(name); }

    void GlRenderer::draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        on_new_frame();

        // on_draw
        const auto camera_pos = camera->pos();
        const glm::mat4 view_matrix = glm::lookAt(camera_pos, camera->look(), camera->up());

        const glm::mat4 proj_matrix = glm::perspective(
            static_cast<float>(M_PI) / 4.f,
            static_cast<float>(get_width()) / static_cast<float>(get_height()), 1.f,
            2000.f * std::sqrt(3.f));

        for (const auto &[name, m_matrix]: model_matrices) {
            auto mv_matrix = view_matrix * m_matrix;
            const auto mvp_matrix = proj_matrix * mv_matrix;

            drawables[name]->draw(mvp_matrix, mv_matrix, light_pos, camera_pos);
        }

        on_end_frame();
    }

    void GlRenderer::make_current() const { gl_context->make_current(); }

    void GlRenderer::release_current() const { gl_context->release_current(); }

    GlRenderer::~GlRenderer() { drawables.clear(); }

    /*
     * GlPlayerRenderer
     */

    GlPlayerRenderer::GlPlayerRenderer(
        const std::shared_ptr<EglRenderContext> &gl_context, const int width, const int height,
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera)
        : GlRenderer(gl_context, light_pos, camera), width(width), height(height), hud_drawables() {
    }

    void GlPlayerRenderer::add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) {
        hud_drawables.push_back(std::move(hud_drawable));
    }

    void GlPlayerRenderer::on_end_frame() {

        glDisable(GL_DEPTH_TEST);

        for (const auto &hud_drawable: hud_drawables) hud_drawable->draw(get_width(), get_height());
        eglSwapBuffers(context()->get_display(), context()->get_surface());

        glEnable(GL_DEPTH_TEST);
    }

    void GlPlayerRenderer::on_new_frame() {
        glViewport(0, 0, get_width(), get_height());

        glClearColor(1., 0., 0., 0.);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);

        glDisable(GL_BLEND);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    GlPlayerRenderer::~GlPlayerRenderer() { hud_drawables.clear(); }

    int GlPlayerRenderer::get_width() const { return width; }

    int GlPlayerRenderer::get_height() const { return height; }

    void GlPlayerRenderer::set_window_size(const int new_width, const int new_height) {
        width = new_width;
        height = new_height;
    }

}// namespace arenai::view
