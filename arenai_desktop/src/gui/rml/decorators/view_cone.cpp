#include "./view_cone.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace arenai::desktop::gui {

    namespace {

        constexpr int NB_SEGMENTS = 16;

    }// namespace

    ViewConeDecorator::ViewConeDecorator(const float span_degrees) : span_degrees_(span_degrees) {}

    Rml::Mesh ViewConeDecorator::build_mesh(Rml::Element &element) const {
        const Rml::Vector2f size = element_size(element);
        const Rml::Vector2f center = size * 0.5f;
        const float radius = 0.5f * std::min(size.x, size.y);
        const float span = span_degrees_ * (std::numbers::pi_v<float> / 180.f);

        Rml::Mesh mesh;
        Rml::Vertex apex;
        apex.position = center;
        apex.colour = element_color(element);
        apex.tex_coord = {0.f, 0.f};
        mesh.vertices.push_back(apex);

        for (int i = 0; i <= NB_SEGMENTS; i++) {
            const float angle =
                (static_cast<float>(i) / static_cast<float>(NB_SEGMENTS) - 0.5f) * span;
            Rml::Vertex rim;
            rim.position = center + Rml::Vector2f(std::sin(angle), -std::cos(angle)) * radius;
            rim.colour = Rml::ColourbPremultiplied(0, 0, 0, 0);
            rim.tex_coord = {0.f, 0.f};
            mesh.vertices.push_back(rim);
        }
        for (int i = 1; i <= NB_SEGMENTS; i++)
            for (const int index: {0, i, i + 1}) mesh.indices.push_back(index);
        return mesh;
    }

    ViewConeDecoratorInstancer::ViewConeDecoratorInstancer() : id_span_(add_number("span", "63")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> ViewConeDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<ViewConeDecorator>(number_of(properties, id_span_));
    }

}// namespace arenai::desktop::gui
