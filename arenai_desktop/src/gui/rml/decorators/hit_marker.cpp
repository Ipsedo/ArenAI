//
// Created by samuel on 23/08/2026.
//

#include "./hit_marker.h"

#include <cmath>

namespace arenai::desktop::gui {

    HitMarkerDecorator::HitMarkerDecorator(
        const Rml::NumericValue thickness, const Rml::NumericValue gap)
        : thickness_(thickness), gap_(gap) {}

    Rml::Mesh HitMarkerDecorator::build_mesh(Rml::Element &element) const {
        const Rml::ColourbPremultiplied color = element_color(element);
        const Rml::Vector2f size = element_size(element);
        const float thickness = element.ResolveLength(thickness_);
        const float gap = element.ResolveLength(gap_);

        const Rml::Vector2f center = size * 0.5f;
        const float half_diagonal = std::sqrt(center.x * center.x + center.y * center.y);
        const float inner_ratio = half_diagonal > 0.f ? gap / half_diagonal : 0.f;

        Rml::Mesh mesh;
        for (const Rml::Vector2f corner:
             {Rml::Vector2f(0.f, 0.f), Rml::Vector2f(size.x, 0.f), Rml::Vector2f(0.f, size.y),
              size}) {
            const Rml::Vector2f inner_end = center + (corner - center) * inner_ratio;
            append_bar(mesh, corner, inner_end, thickness, color);
        }
        return mesh;
    }

    HitMarkerDecoratorInstancer::HitMarkerDecoratorInstancer()
        : id_thickness_(add_length("thickness", "3dp")), id_gap_(add_length("gap", "6dp")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> HitMarkerDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<HitMarkerDecorator>(
            length_of(properties, id_thickness_), length_of(properties, id_gap_));
    }

}// namespace arenai::desktop::gui
