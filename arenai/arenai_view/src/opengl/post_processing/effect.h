//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_EFFECT_H
#define ARENAI_POST_PROCESSING_EFFECT_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <GLES3/gl3.h>
#include <glm/glm.hpp>

#include "../program.h"

namespace arenai::view {

    // One pass (or ping-pong group of passes) of the post-processing chain.
    // The concrete effect declares its render targets (format + resolution
    // divisor) at construction; this base class owns their lifecycle and the
    // fullscreen-triangle draw. Effects communicate through the FrameContext:
    // each pass publishes its output texture under a name that later passes
    // look up.
    class AbstractPostProcessingEffect {
    public:
        // per-frame inputs shared by every pass, plus the outputs published
        // by the passes already run this frame
        struct FrameContext {
            GLuint scene_texture;// resolved scene color
            GLuint depth_texture;// resolved scene depth
            int screen_width;
            int screen_height;
            glm::mat4 proj_matrix;
            glm::vec4 proj_info;
            glm::vec3 sun_dir_view;
            int frame;

            std::unordered_map<std::string, GLuint> textures;
            std::unordered_map<std::string, float> scalars;
        };

        virtual ~AbstractPostProcessingEffect();

        // recreates the effect's targets at the new screen resolution
        void resize(int new_width, int new_height);

        // binds uniforms/textures, draws the pass(es) and publishes the
        // effect's outputs into the context
        virtual void render(FrameContext &context) = 0;

    protected:
        // offscreen render target (single color attachment)
        struct Target {
            GLuint fbo = 0;
            GLuint texture = 0;
            int width = 0;
            int height = 0;
        };

        // declares one target: screen resolution / size_divisor
        struct TargetSpec {
            GLenum internal_format;
            int size_divisor;
        };

        AbstractPostProcessingEffect(
            std::unique_ptr<Program> program, std::vector<TargetSpec> specs, int width, int height);

        // shared Program::Builder for a fullscreen-triangle effect shader
        static Program::Builder effect_builder(const char *fragment_shader);

        // runs the current program over a fullscreen triangle into fbo; a
        // null fbo with the screen size targets the default framebuffer
        void draw_fullscreen(GLuint fbo, int viewport_width, int viewport_height) const;

        std::unique_ptr<Program> program;
        // created from specs, same order (empty for a to-screen effect)
        std::vector<Target> targets;

    private:
        static Target create_target(GLenum internal_format, int width, int height);

        void create_targets(int width, int height);
        void destroy_targets();

        std::vector<TargetSpec> specs;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_EFFECT_H
