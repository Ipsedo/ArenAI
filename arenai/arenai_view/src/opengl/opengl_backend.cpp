//
// Created by samuel on 08/07/2026.
//

#include "./opengl_backend.h"

#include <string>
#include <utility>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include "./renderers/gl_offscreen_renderer.h"

namespace arenai::view {

    /*
     * OpenGlBackend (headless)
     */

    OpenGlBackend::OpenGlBackend(std::shared_ptr<EglRenderContext> context)
        : context_(std::move(context)) {}

    std::shared_ptr<AbstractRenderContext> OpenGlBackend::render_context() { return context_; }

    std::unique_ptr<AbstractOffscreenRenderer> OpenGlBackend::make_offscreen_renderer(
        const int width, const int height, const glm::vec3 light_pos,
        const std::shared_ptr<AbstractCamera> &camera) {
        return std::make_unique<GlOffscreenRenderer>(context_, width, height, light_pos, camera);
    }

    std::shared_ptr<AbstractDrawableFactory> OpenGlBackend::drawable_factory() {
        return drawable_factory_;
    }

    std::shared_ptr<AbstractHudFactory> OpenGlBackend::hud_factory() { return hud_factory_; }

    std::string OpenGlBackend::renderer_info() {
        context_->make_current();
        const auto query = [](const GLenum name) {
            const auto *value = glGetString(name);
            return value ? std::string(reinterpret_cast<const char *>(value))
                         : std::string("unknown");
        };
        return "vendor=" + query(GL_VENDOR) + ", renderer=" + query(GL_RENDERER)
               + ", version=" + query(GL_VERSION);
    }

    void OpenGlBackend::release_thread() { eglReleaseThread(); }

    /*
     * OpenGlViewFactory (headless part; windowed part in src/glfw)
     */

    std::unique_ptr<AbstractGraphicBackend> make_opengl_backend() {
        return std::make_unique<OpenGlBackend>(std::make_shared<HeadlessEglContext>());
    }

}// namespace arenai::view
