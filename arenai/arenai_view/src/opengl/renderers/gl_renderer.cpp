//
// Created by samuel on 29/09/2025.
//

#include "./gl_renderer.h"

#include <cmath>
#include <utility>

#include <GLES3/gl3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "../drawables/shadow_drawable.h"

using namespace arenai;

namespace arenai::view {

    /*
     * GlRenderer
     */

    // maps light-space NDC [-1, 1] to shadow-map coordinates [0, 1]
    constexpr glm::mat4 SHADOW_BIAS_MATRIX(
        0.5f, 0.f, 0.f, 0.f, 0.f, 0.5f, 0.f, 0.f, 0.f, 0.f, 0.5f, 0.f, 0.5f, 0.5f, 0.5f, 1.f);

    GlRenderer::GlRenderer(
        std::shared_ptr<EglRenderContext> gl_context, const glm::vec3 light_pos,
        std::shared_ptr<AbstractCamera> camera, const bool with_shadows)
        : light_pos(light_pos), with_shadows(with_shadows), gl_context(std::move(gl_context)),
          camera(std::move(camera)) {}

    void
    GlRenderer::add_drawable(const std::string &name, std::unique_ptr<AbstractDrawable> drawable) {
        drawables.insert({name, std::move(drawable)});
    }

    void GlRenderer::remove_drawable(const std::string &name) { drawables.erase(name); }

    glm::mat4 GlRenderer::light_view_projection() const {
        const glm::vec3 light_dir = glm::normalize(light_pos);
        const auto up =
            std::abs(light_dir.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);

        const glm::mat4 light_view = glm::lookAt(light_dir * SHADOW_DISTANCE, glm::vec3(0.f), up);

        // center the ortho frustum on the camera, snapped to the shadow-map
        // texel grid to avoid shadow shimmering when the camera moves
        const auto center = glm::vec3(light_view * glm::vec4(camera->pos(), 1.f));
        constexpr float texel_size = 2.f * SHADOW_HALF_EXTENT / static_cast<float>(SHADOW_MAP_SIZE);
        const float x = std::floor(center.x / texel_size) * texel_size;
        const float y = std::floor(center.y / texel_size) * texel_size;
        const float depth = -center.z;

        const glm::mat4 light_proj = glm::ortho(
            x - SHADOW_HALF_EXTENT, x + SHADOW_HALF_EXTENT, y - SHADOW_HALF_EXTENT,
            y + SHADOW_HALF_EXTENT, depth - SHADOW_DEPTH_RANGE, depth + SHADOW_DEPTH_RANGE);

        return light_proj * light_view;
    }

    void GlRenderer::draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        glm::mat4 light_vp_matrix(1.f);
        bool shadow_pass_done = false;

        if (with_shadows) {
            if (!shadow_map) shadow_map = std::make_unique<ShadowMap>(SHADOW_MAP_SIZE);

            light_vp_matrix = light_view_projection();

            shadow_map->begin_depth_pass();
            for (const auto &[name, m_matrix]: model_matrices)
                if (auto *shadow_drawable = dynamic_cast<GlShadowDrawable *>(drawables[name].get()))
                    shadow_drawable->draw_depth(light_vp_matrix * m_matrix);
            shadow_map->end_depth_pass();

            shadow_pass_done = true;
        }

        on_new_frame();

        // on_draw
        const auto camera_pos = camera->pos();
        const glm::mat4 view_matrix = glm::lookAt(camera_pos, camera->look(), camera->up());

        const glm::mat4 proj_matrix = glm::perspective(
            static_cast<float>(M_PI) / 4.f,
            static_cast<float>(get_width()) / static_cast<float>(get_height()), 1.f,
            2000.f * std::sqrt(3.f));

        const glm::mat4 biased_light_vp_matrix = SHADOW_BIAS_MATRIX * light_vp_matrix;

        // world up axis in view space (xyz) + camera world height (w): enough
        // for the shadow shader to do hemisphere ambient and height fog
        // without a full inverse view matrix
        const glm::vec4 world_up(glm::mat3(view_matrix) * glm::vec3(0.f, 1.f, 0.f), camera_pos.y);

        for (const auto &[name, m_matrix]: model_matrices) {
            auto mv_matrix = view_matrix * m_matrix;
            const auto mvp_matrix = proj_matrix * mv_matrix;

            auto *shadow_drawable = shadow_pass_done
                                        ? dynamic_cast<GlShadowDrawable *>(drawables[name].get())
                                        : nullptr;

            if (shadow_drawable)
                shadow_drawable->draw_with_shadow(
                    mvp_matrix, mv_matrix, light_pos, camera_pos, world_up,
                    biased_light_vp_matrix * m_matrix, shadow_map->depth_texture());
            else drawables[name]->draw(mvp_matrix, mv_matrix, light_pos, camera_pos);
        }

        on_end_frame(view_matrix, proj_matrix);
    }

    void GlRenderer::make_current() const { gl_context->make_current(); }

    void GlRenderer::release_current() const { gl_context->release_current(); }

    const std::shared_ptr<EglRenderContext> &GlRenderer::context() const { return gl_context; }

    const glm::vec3 &GlRenderer::light_position() const { return light_pos; }

    GlRenderer::~GlRenderer() { drawables.clear(); }

    /*
     * GlPlayerRenderer
     */

    GlPlayerRenderer::GlPlayerRenderer(
        const std::shared_ptr<EglRenderContext> &gl_context, const int width, const int height,
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera)
        : GlRenderer(gl_context, light_pos, camera, true), width(width), height(height),
          hud_drawables() {}

    void GlPlayerRenderer::add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) {
        hud_drawables.push_back(std::move(hud_drawable));
    }

    void
    GlPlayerRenderer::on_end_frame(const glm::mat4 &view_matrix, const glm::mat4 &proj_matrix) {
        // post-processing pass onto the default framebuffer, then the HUD on
        // top so that it stays untouched by the effects
        const glm::vec3 sun_dir_view =
            glm::normalize(glm::mat3(view_matrix) * glm::normalize(light_position()));
        post_process->draw_to_screen(proj_matrix, sun_dir_view);

        glDisable(GL_DEPTH_TEST);

        for (const auto &hud_drawable: hud_drawables) hud_drawable->draw(get_width(), get_height());
        eglSwapBuffers(context()->get_display(), context()->get_surface());

        glEnable(GL_DEPTH_TEST);
    }

    void GlPlayerRenderer::on_new_frame() {
        if (!post_process)
            post_process = std::make_unique<PostProcess>(
                get_width(), get_height(),
                make_default_post_processing_effects(get_width(), get_height()));
        post_process->begin_scene_pass();

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

        glViewport(0, 0, width, height);

        if (post_process) post_process->resize(width, height);
    }

}// namespace arenai::view
