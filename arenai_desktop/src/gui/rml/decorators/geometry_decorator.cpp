//
// Created by samuel on 25/10/2026.
//

#include "./geometry_decorator.h"

#include <cmath>

namespace arenai::desktop::gui {

    Rml::DecoratorDataHandle
    GeometryDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(build_mesh(*element)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void GeometryDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void GeometryDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    Rml::ColourbPremultiplied GeometryDecorator::element_color(const Rml::Element &element) {
        const auto &computed = element.GetComputedValues();
        return computed.color().ToPremultiplied(computed.opacity());
    }

    Rml::Vector2f GeometryDecorator::element_size(Rml::Element &element) {
        return element.GetBox().GetSize(Rml::BoxArea::Border);
    }

    void append_bar(
        Rml::Mesh &mesh, const Rml::Vector2f from, const Rml::Vector2f to, const float thickness,
        const Rml::ColourbPremultiplied color) {
        const Rml::Vector2f direction = to - from;
        const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
        if (length <= 0.f) return;
        const Rml::Vector2f normal =
            Rml::Vector2f(-direction.y, direction.x) * (0.5f * thickness / length);

        const int base = static_cast<int>(mesh.vertices.size());
        for (const Rml::Vector2f position:
             {from + normal, to + normal, to - normal, from - normal}) {
            Rml::Vertex vertex;
            vertex.position = position;
            vertex.colour = color;
            vertex.tex_coord = {0.f, 0.f};
            mesh.vertices.push_back(vertex);
        }
        for (const int index: {0, 1, 2, 0, 2, 3}) mesh.indices.push_back(base + index);
    }

    Rml::PropertyId GeometryDecoratorInstancer::add(
        const char *name, const char *default_value, const char *parser) {
        property_names_.emplace_back(name);
        return RegisterProperty(name, default_value).AddParser(parser).GetId();
    }

    Rml::PropertyId
    GeometryDecoratorInstancer::add_length(const char *name, const char *default_value) {
        return add(name, default_value, "length");
    }

    Rml::PropertyId
    GeometryDecoratorInstancer::add_color(const char *name, const char *default_value) {
        return add(name, default_value, "color");
    }

    Rml::PropertyId
    GeometryDecoratorInstancer::add_number(const char *name, const char *default_value) {
        return add(name, default_value, "number");
    }

    void GeometryDecoratorInstancer::register_shorthand() {
        Rml::String names;
        for (const auto &name: property_names_) names += (names.empty() ? "" : ", ") + name;
        RegisterShorthand("decorator", names, Rml::ShorthandType::FallThrough);
    }

    Rml::NumericValue GeometryDecoratorInstancer::length_of(
        const Rml::PropertyDictionary &properties, const Rml::PropertyId id) {
        return properties.GetProperty(id)->GetNumericValue();
    }

    Rml::Colourb GeometryDecoratorInstancer::color_of(
        const Rml::PropertyDictionary &properties, const Rml::PropertyId id) {
        return properties.GetProperty(id)->Get<Rml::Colourb>();
    }

    float GeometryDecoratorInstancer::number_of(
        const Rml::PropertyDictionary &properties, const Rml::PropertyId id) {
        return properties.GetProperty(id)->Get<float>();
    }

}// namespace arenai::desktop::gui
