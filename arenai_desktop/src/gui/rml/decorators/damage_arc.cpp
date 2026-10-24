//
// Created by samuel on 27/08/2026.
//

#include "./damage_arc.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace arenai::desktop::gui {

    namespace {

        constexpr int NB_SEGMENTS = 24;

        Rml::ColourbPremultiplied
        scale_color(const Rml::ColourbPremultiplied color, const float factor) {
            return {
                static_cast<Rml::byte>(static_cast<float>(color.red) * factor),
                static_cast<Rml::byte>(static_cast<float>(color.green) * factor),
                static_cast<Rml::byte>(static_cast<float>(color.blue) * factor),
                static_cast<Rml::byte>(static_cast<float>(color.alpha) * factor)};
        }

    }// namespace

    DamageArcDecorator::DamageArcDecorator(
        const Rml::NumericValue thickness, const float span_degrees)
        : thickness_(thickness), span_degrees_(span_degrees) {}

    Rml::Mesh DamageArcDecorator::build_mesh(Rml::Element &element) const {
        const Rml::ColourbPremultiplied color = element_color(element);
        const Rml::Vector2f size = element_size(element);
        const float thickness = element.ResolveLength(thickness_);

        const Rml::Vector2f center = size * 0.5f;
        const float outer_radius = 0.5f * std::min(size.x, size.y);
        const float inner_radius = std::max(outer_radius - thickness, 0.f);
        const float span = span_degrees_ * (std::numbers::pi_v<float> / 180.f);

        // the color fades quadratically toward both tips of the arc
        Rml::Mesh mesh;
        for (int i = 0; i <= NB_SEGMENTS; i++) {
            const float t = static_cast<float>(i) / static_cast<float>(NB_SEGMENTS);
            const float angle = (t - 0.5f) * span;
            const Rml::Vector2f direction(std::sin(angle), -std::cos(angle));

            const float tip_fade = 1.f - (2.f * t - 1.f) * (2.f * t - 1.f);
            const Rml::ColourbPremultiplied faded = scale_color(color, tip_fade);

            for (const float radius: {inner_radius, outer_radius}) {
                Rml::Vertex vertex;
                vertex.position = center + direction * radius;
                vertex.colour = faded;
                vertex.tex_coord = {0.f, 0.f};
                mesh.vertices.push_back(vertex);
            }
        }
        for (int i = 0; i < NB_SEGMENTS; i++) {
            const int base = 2 * i;
            for (const int index: {0, 1, 3, 0, 3, 2}) mesh.indices.push_back(base + index);
        }
        return mesh;
    }

    DamageArcDecoratorInstancer::DamageArcDecoratorInstancer()
        : id_thickness_(add_length("thickness", "16dp")), id_span_(add_number("span", "60")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> DamageArcDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<DamageArcDecorator>(
            length_of(properties, id_thickness_), number_of(properties, id_span_));
    }

}// namespace arenai::desktop::gui
