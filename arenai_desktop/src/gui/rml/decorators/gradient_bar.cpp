//
// Created by samuel on 28/10/2026.
//

#include "./gradient_bar.h"

#include <algorithm>
#include <array>

namespace arenai::desktop::gui {

    namespace {

        // a slice of the bar, its left and right edges each carrying their
        // own color (the GPU interpolates between them)
        void append_slice(
            Rml::Mesh &mesh, const float x0, const float x1, const float height,
            const Rml::ColourbPremultiplied left, const Rml::ColourbPremultiplied right) {
            if (x1 <= x0) return;
            const int base = static_cast<int>(mesh.vertices.size());
            const std::array<Rml::Vertex, 4> corners = {
                Rml::Vertex{{x0, 0.f}, left, {0.f, 0.f}}, Rml::Vertex{{x1, 0.f}, right, {0.f, 0.f}},
                Rml::Vertex{{x1, height}, right, {0.f, 0.f}},
                Rml::Vertex{{x0, height}, left, {0.f, 0.f}}};
            for (const Rml::Vertex &vertex: corners) mesh.vertices.push_back(vertex);
            for (const int index: {0, 1, 2, 0, 2, 3}) mesh.indices.push_back(base + index);
        }

    }// namespace

    GradientBarDecorator::GradientBarDecorator(
        const Rml::Colourb left_color, const Rml::Colourb right_color, const float fade_percent)
        : left_color_(left_color), right_color_(right_color), fade_percent_(fade_percent) {}

    Rml::Mesh GradientBarDecorator::build_mesh(Rml::Element &element) const {
        const float opacity = element.GetComputedValues().opacity();
        const Rml::Vector2f size = element_size(element);
        const float fade = std::clamp(fade_percent_, 0.f, 50.f) * 0.01f * size.x;
        const Rml::ColourbPremultiplied left = left_color_.ToPremultiplied(opacity);
        const Rml::ColourbPremultiplied right = right_color_.ToPremultiplied(opacity);
        const Rml::ColourbPremultiplied clear(0, 0, 0, 0);

        Rml::Mesh mesh;
        append_slice(mesh, 0.f, fade, size.y, clear, left);
        append_slice(mesh, fade, size.x - fade, size.y, left, right);
        append_slice(mesh, size.x - fade, size.x, size.y, right, clear);
        return mesh;
    }

    GradientBarDecoratorInstancer::GradientBarDecoratorInstancer()
        : id_left_color_(add_color("left-color", "white")),
          id_right_color_(add_color("right-color", "white")), id_fade_(add_number("fade", "30")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> GradientBarDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<GradientBarDecorator>(
            color_of(properties, id_left_color_), color_of(properties, id_right_color_),
            number_of(properties, id_fade_));
    }

}// namespace arenai::desktop::gui
