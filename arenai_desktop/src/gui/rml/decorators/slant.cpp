//
// Created by samuel on 27/10/2026.
//

#include "./slant.h"

#include <algorithm>
#include <array>

namespace arenai::desktop::gui {

    namespace {

        void append_quad(
            Rml::Mesh &mesh, const std::array<Rml::Vector2f, 4> &corners,
            const Rml::ColourbPremultiplied color) {
            const int base = static_cast<int>(mesh.vertices.size());
            for (const Rml::Vector2f position: corners) {
                Rml::Vertex vertex;
                vertex.position = position;
                vertex.colour = color;
                vertex.tex_coord = {0.f, 0.f};
                mesh.vertices.push_back(vertex);
            }
            for (const int index: {0, 1, 2, 0, 2, 3}) mesh.indices.push_back(base + index);
        }

    }// namespace

    SlantDecorator::SlantDecorator(
        const Rml::Colourb fill_color, const Rml::Colourb line_color, const Rml::NumericValue slant,
        const Rml::NumericValue thickness)
        : fill_color_(fill_color), line_color_(line_color), slant_(slant), thickness_(thickness) {}

    Rml::Mesh SlantDecorator::build_mesh(Rml::Element &element) const {
        const float opacity = element.GetComputedValues().opacity();
        const Rml::Vector2f size = element_size(element);
        // the lean can never exceed what the width leaves for the top edge
        const float slant = std::clamp(element.ResolveLength(slant_), 0.f, size.x * 0.5f);
        const float thickness = element.ResolveLength(thickness_);

        Rml::Mesh mesh;

        // top-left, top-right, bottom-right, bottom-left
        const std::array<Rml::Vector2f, 4> outer = {
            Rml::Vector2f(slant, 0.f), Rml::Vector2f(size.x, 0.f),
            Rml::Vector2f(size.x - slant, size.y), Rml::Vector2f(0.f, size.y)};

        if (fill_color_.alpha > 0) append_quad(mesh, outer, fill_color_.ToPremultiplied(opacity));

        // outline: one bar per edge, centered half a thickness inside the
        // shape so nothing spills out of the border box
        if (line_color_.alpha > 0 && thickness > 0.f) {
            const Rml::ColourbPremultiplied line = line_color_.ToPremultiplied(opacity);
            const float half = 0.5f * thickness;
            // the leaning edges keep the same horizontal inset as the lean's
            // slope demands, approximated by the plain half thickness
            const std::array<Rml::Vector2f, 4> inner = {
                Rml::Vector2f(slant + half, half), Rml::Vector2f(size.x - half, half),
                Rml::Vector2f(size.x - slant - half, size.y - half),
                Rml::Vector2f(half, size.y - half)};
            for (std::size_t i = 0; i < inner.size(); ++i)
                append_bar(mesh, inner[i], inner[(i + 1) % inner.size()], thickness, line);
        }

        return mesh;
    }

    SlantDecoratorInstancer::SlantDecoratorInstancer()
        : id_fill_color_(add_color("fill-color", "transparent")),
          id_line_color_(add_color("line-color", "transparent")),
          id_slant_(add_length("slant", "12dp")), id_thickness_(add_length("thickness", "1dp")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> SlantDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<SlantDecorator>(
            color_of(properties, id_fill_color_), color_of(properties, id_line_color_),
            length_of(properties, id_slant_), length_of(properties, id_thickness_));
    }

}// namespace arenai::desktop::gui
