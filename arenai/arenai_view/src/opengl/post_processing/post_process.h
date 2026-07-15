//
// Created by samuel on 14/07/2026.
//

#ifndef ARENAI_POST_PROCESS_H
#define ARENAI_POST_PROCESS_H

#include <memory>
#include <vector>

#include <GLES3/gl3.h>
#include <glm/glm.hpp>

#include "./effect.h"

namespace arenai::view {

    // Player-only post-processing pipeline: the scene is rendered into a 4x
    // MSAA framebuffer, resolved (color + depth), then run through the
    // ordered effect chain, whose last effect draws onto the default
    // framebuffer. Must be created and used with the target GL context
    // current.
    class PostProcess {
    public:
        PostProcess(
            int width, int height,
            std::vector<std::shared_ptr<AbstractPostProcessingEffect>> ordered_effects);
        ~PostProcess();

        PostProcess(const PostProcess &) = delete;
        PostProcess &operator=(const PostProcess &) = delete;

        void resize(int new_width, int new_height);

        // binds the MSAA framebuffer: subsequent scene draws render into it
        void begin_scene_pass() const;

        // runs all the effect passes and draws the post-processed frame onto
        // the default framebuffer, leaving it bound (for the HUD and the
        // swap); proj_matrix is the scene projection (depth reconstruction)
        // and sun_dir_view the normalized view-space direction toward the sun
        void draw_to_screen(const glm::mat4 &proj_matrix, const glm::vec3 &sun_dir_view);

    private:
        static constexpr int MSAA_SAMPLES = 4;

        void create_scene_targets();
        void destroy_scene_targets();

        int width;
        int height;

        // frame counter animating the film grain (wraps to stay float-exact)
        int frame;

        // scene render target (MSAA) and its single-sampled resolve — the
        // input of the effect chain, owned here because it is not an effect
        GLuint msaa_fbo;
        GLuint msaa_color_rbo;
        GLuint msaa_depth_rbo;

        GLuint resolve_fbo;
        GLuint resolve_texture;
        GLuint resolve_depth_texture;

        std::vector<std::shared_ptr<AbstractPostProcessingEffect>> ordered_post_processing_effects;
    };

    // the standard player chain: SSAO → AO blur → bloom bright → bloom blur
    // → god rays → composite
    std::vector<std::shared_ptr<AbstractPostProcessingEffect>>
    make_default_post_processing_effects(int width, int height);

}// namespace arenai::view

#endif// ARENAI_POST_PROCESS_H
