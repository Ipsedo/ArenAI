//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_RML_RENDER_INTERFACE_H
#define ARENAI_RML_RENDER_INTERFACE_H

#include <RmlUi/Core/RenderInterface.h>

#include "./gl.h"

namespace arenai::view {

    // OpenGL 3.3 core adapter behind RmlUi's abstract render interface. This is
    // the only place where the UI library touches the GL API: the rest of the
    // application only sees the forward-declared Rml::RenderInterface handed
    // out by AbstractWindowedGraphicBackend.
    class RmlGlRenderInterface final : public Rml::RenderInterface {
    public:
        RmlGlRenderInterface() = default;
        ~RmlGlRenderInterface() override;

        // called by the backend around Rml::Context::Render(), with the window
        // context current: sets the 2D projection and the UI blending state
        void begin_frame(int viewport_width, int viewport_height);
        void end_frame();

        Rml::CompiledGeometryHandle CompileGeometry(
            Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
        void RenderGeometry(
            Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
            Rml::TextureHandle texture) override;
        void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

        Rml::TextureHandle
        LoadTexture(Rml::Vector2i &texture_dimensions, const Rml::String &source) override;
        Rml::TextureHandle GenerateTexture(
            Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) override;
        void ReleaseTexture(Rml::TextureHandle texture) override;

        void EnableScissorRegion(bool enable) override;
        void SetScissorRegion(Rml::Rectanglei region) override;

    private:
        struct CompiledGeometry {
            GLuint vao;
            GLuint vbo;
            GLuint ibo;
            GLsizei nb_indices;
        };

        // GL objects are created on first begin_frame(), when a context is
        // guaranteed to be current
        void lazy_init();

        bool initialized_ = false;
        GLuint program_ = 0;
        GLint u_projection_ = -1;
        GLint u_translation_ = -1;
        GLint a_position_ = -1;
        GLint a_color_ = -1;
        GLint a_tex_coord_ = -1;

        // bound for untextured geometry so that one program handles both cases
        GLuint white_texture_ = 0;

        int viewport_height_ = 0;
    };

}// namespace arenai::view

#endif// ARENAI_RML_RENDER_INTERFACE_H
