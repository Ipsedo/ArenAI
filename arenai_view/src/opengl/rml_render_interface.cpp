//
// Created by samuel on 17/07/2026.
//

#include "./rml_render_interface.h"

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "./shader.h"

namespace arenai::view {

    void RmlGlRenderInterface::lazy_init() {
        if (initialized_) return;

        const GLuint vertex_shader = load_shader(GL_VERTEX_SHADER, "rml_vs.glsl");
        const GLuint fragment_shader = load_shader(GL_FRAGMENT_SHADER, "rml_fs.glsl");

        program_ = glCreateProgram();
        glAttachShader(program_, vertex_shader);
        glAttachShader(program_, fragment_shader);
        glLinkProgram(program_);

        GLint linked = GL_FALSE;
        glGetProgramiv(program_, GL_LINK_STATUS, &linked);
        if (!linked) throw std::runtime_error("RmlUi GL program link failed");

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        u_projection_ = glGetUniformLocation(program_, "u_projection");
        u_translation_ = glGetUniformLocation(program_, "u_translation");
        a_position_ = glGetAttribLocation(program_, "a_position");
        a_color_ = glGetAttribLocation(program_, "a_color");
        a_tex_coord_ = glGetAttribLocation(program_, "a_tex_coord");

        constexpr uint8_t white[4] = {255, 255, 255, 255};
        glGenTextures(1, &white_texture_);
        glBindTexture(GL_TEXTURE_2D, white_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        initialized_ = true;
    }

    RmlGlRenderInterface::~RmlGlRenderInterface() {
        if (program_) glDeleteProgram(program_);
        if (white_texture_) glDeleteTextures(1, &white_texture_);
    }

    void RmlGlRenderInterface::begin_frame(const int viewport_width, const int viewport_height) {
        lazy_init();

        viewport_height_ = viewport_height;

        glUseProgram(program_);

        // pixel coordinates with the origin at the top-left, as RmlUi expects
        const glm::mat4 projection = glm::ortho(
            0.f, static_cast<float>(viewport_width), static_cast<float>(viewport_height), 0.f);
        glUniformMatrix4fv(u_projection_, 1, GL_FALSE, glm::value_ptr(projection));

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);

        // RmlUi outputs premultiplied-alpha colors
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
    }

    void RmlGlRenderInterface::end_frame() {
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    Rml::CompiledGeometryHandle RmlGlRenderInterface::CompileGeometry(
        const Rml::Span<const Rml::Vertex> vertices, const Rml::Span<const int> indices) {
        auto *geometry = new CompiledGeometry{};
        geometry->nb_indices = static_cast<GLsizei>(indices.size());

        glGenVertexArrays(1, &geometry->vao);
        glGenBuffers(1, &geometry->vbo);
        glGenBuffers(1, &geometry->ibo);

        glBindVertexArray(geometry->vao);

        glBindBuffer(GL_ARRAY_BUFFER, geometry->vbo);
        glBufferData(
            GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Rml::Vertex)),
            vertices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(a_position_);
        glVertexAttribPointer(
            a_position_, 2, GL_FLOAT, GL_FALSE, sizeof(Rml::Vertex),
            reinterpret_cast<const void *>(offsetof(Rml::Vertex, position)));
        glEnableVertexAttribArray(a_color_);
        glVertexAttribPointer(
            a_color_, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Rml::Vertex),
            reinterpret_cast<const void *>(offsetof(Rml::Vertex, colour)));
        glEnableVertexAttribArray(a_tex_coord_);
        glVertexAttribPointer(
            a_tex_coord_, 2, GL_FLOAT, GL_FALSE, sizeof(Rml::Vertex),
            reinterpret_cast<const void *>(offsetof(Rml::Vertex, tex_coord)));

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geometry->ibo);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(int)),
            indices.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);

        return reinterpret_cast<Rml::CompiledGeometryHandle>(geometry);
    }

    void RmlGlRenderInterface::RenderGeometry(
        const Rml::CompiledGeometryHandle geometry, const Rml::Vector2f translation,
        const Rml::TextureHandle texture) {
        const auto *compiled = reinterpret_cast<const CompiledGeometry *>(geometry);

        glUniform2f(u_translation_, translation.x, translation.y);
        glBindTexture(GL_TEXTURE_2D, texture ? static_cast<GLuint>(texture) : white_texture_);

        glBindVertexArray(compiled->vao);
        glDrawElements(GL_TRIANGLES, compiled->nb_indices, GL_UNSIGNED_INT, nullptr);
    }

    void RmlGlRenderInterface::ReleaseGeometry(const Rml::CompiledGeometryHandle geometry) {
        const auto *compiled = reinterpret_cast<const CompiledGeometry *>(geometry);

        glDeleteVertexArrays(1, &compiled->vao);
        glDeleteBuffers(1, &compiled->vbo);
        glDeleteBuffers(1, &compiled->ibo);

        delete compiled;
    }

    Rml::TextureHandle RmlGlRenderInterface::LoadTexture(
        Rml::Vector2i &texture_dimensions, const Rml::String &source) {
        // no image decoding on the UI path yet: the menu documents are styled
        // with plain decorators (colors, borders) and need no texture files
        std::cerr << "RmlUi texture files are not supported (requested: " << source << ")"
                  << std::endl;
        (void) texture_dimensions;
        return 0;
    }

    Rml::TextureHandle RmlGlRenderInterface::GenerateTexture(
        const Rml::Span<const Rml::byte> source, const Rml::Vector2i source_dimensions) {
        GLuint texture_id = 0;
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8, source_dimensions.x, source_dimensions.y, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, source.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        return static_cast<Rml::TextureHandle>(texture_id);
    }

    void RmlGlRenderInterface::ReleaseTexture(const Rml::TextureHandle texture) {
        auto texture_id = static_cast<GLuint>(texture);
        glDeleteTextures(1, &texture_id);
    }

    void RmlGlRenderInterface::EnableScissorRegion(const bool enable) {
        if (enable) glEnable(GL_SCISSOR_TEST);
        else glDisable(GL_SCISSOR_TEST);
    }

    void RmlGlRenderInterface::SetScissorRegion(const Rml::Rectanglei region) {
        // RmlUi regions are top-left based, glScissor is bottom-left based
        glScissor(
            region.Left(), viewport_height_ - region.Bottom(), region.Width(), region.Height());
    }

}// namespace arenai::view
