//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_GL_RENDERER_H
#define ARENAI_GL_RENDERER_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_view/camera.h>
#include <arenai_view/renderer.h>

#include "../egl_render_context.h"
#include "../shadow_map.h"

namespace arenai::view {

    class GlRenderer : public virtual AbstractRenderer {
    public:
        GlRenderer(
            std::shared_ptr<EglRenderContext> gl_context, glm::vec3 light_pos,
            std::shared_ptr<AbstractCamera> camera, bool with_shadows);
        ~GlRenderer() override;

        void
        add_drawable(const std::string &name, std::unique_ptr<AbstractDrawable> drawable) override;
        void remove_drawable(const std::string &name) override;

        void draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void make_current() const override;
        void release_current() const override;

    protected:
        virtual void on_new_frame() = 0;
        virtual void on_end_frame() = 0;

        const std::shared_ptr<EglRenderContext> &context() const;

    private:
        static constexpr int SHADOW_MAP_SIZE = 16384;
        // ortho frustum half extent, centered on the camera (the arena is far
        // too large to be covered by a single shadow map at a usable resolution)
        static constexpr float SHADOW_HALF_EXTENT = 500.f;
        static constexpr float SHADOW_DISTANCE = 1000.f;
        // must cover the light-space depth spread of the whole frustum: with a
        // ~47° light elevation, ground at the frustum corners reaches ~±500
        static constexpr float SHADOW_DEPTH_RANGE = 900.f;

        glm::mat4 light_view_projection() const;

        glm::vec3 light_pos;

        bool with_shadows;
        std::unique_ptr<ShadowMap> shadow_map;

        std::map<std::string, std::unique_ptr<AbstractDrawable>> drawables;

        std::shared_ptr<EglRenderContext> gl_context;

        std::shared_ptr<AbstractCamera> camera;
    };

    class GlPlayerRenderer final : public GlRenderer, public AbstractPlayerRenderer {
    public:
        GlPlayerRenderer(
            const std::shared_ptr<EglRenderContext> &gl_context, int width, int height,
            glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera);

        void add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) override;

        int get_width() const override;
        int get_height() const override;

        void set_window_size(int new_width, int new_height) override;

        ~GlPlayerRenderer() override;

    protected:
        void on_new_frame() override;
        void on_end_frame() override;

    private:
        int width;
        int height;

        std::vector<std::unique_ptr<AbstractHudDrawable>> hud_drawables;
    };

}// namespace arenai::view

#endif// ARENAI_GL_RENDERER_H
