#include "./chevron.h"

#include <algorithm>

namespace arenai::desktop::gui {

    namespace {

        // tip, right wing, notch, left wing; fractions of the half size around the center
        constexpr float TIP_Y = -1.f;
        constexpr float WING_X = 0.72f;
        constexpr float WING_Y = 0.8f;
        constexpr float NOTCH_Y = 0.38f;

        void append_chevron(
            Rml::Mesh &mesh, const Rml::Vector2f center, const float half_size,
            const Rml::ColourbPremultiplied color) {
            const int base = static_cast<int>(mesh.vertices.size());
            for (const Rml::Vector2f point:
                 {Rml::Vector2f(0.f, TIP_Y), Rml::Vector2f(WING_X, WING_Y),
                  Rml::Vector2f(0.f, NOTCH_Y), Rml::Vector2f(-WING_X, WING_Y)}) {
                Rml::Vertex vertex;
                vertex.position = center + point * half_size;
                vertex.colour = color;
                vertex.tex_coord = {0.f, 0.f};
                mesh.vertices.push_back(vertex);
            }
            for (const int index: {0, 1, 2, 0, 2, 3}) mesh.indices.push_back(base + index);
        }

    }// namespace

    ChevronDecorator::ChevronDecorator(
        const Rml::Colourb outline_color, const Rml::NumericValue outline_width)
        : outline_color_(outline_color), outline_width_(outline_width) {}

    Rml::Mesh ChevronDecorator::build_mesh(Rml::Element &element) const {
        const Rml::Vector2f size = element_size(element);
        const float opacity = element.GetComputedValues().opacity();
        const float outline = element.ResolveLength(outline_width_);

        const float half_size = 0.5f * std::min(size.x, size.y) - outline;
        const Rml::Vector2f center(0.5f * size.x, 0.5f * size.y + 0.1f * half_size);

        Rml::Mesh mesh;
        append_chevron(mesh, center, half_size + outline, outline_color_.ToPremultiplied(opacity));
        append_chevron(mesh, center, half_size, element_color(element));
        return mesh;
    }

    ChevronDecoratorInstancer::ChevronDecoratorInstancer()
        : id_outline_color_(add_color("outline-color", "#0D0B1A")),
          id_outline_width_(add_length("outline-width", "1.5dp")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> ChevronDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<ChevronDecorator>(
            color_of(properties, id_outline_color_), length_of(properties, id_outline_width_));
    }

}// namespace arenai::desktop::gui
