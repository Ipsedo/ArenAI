//
// Created by samuel on 23/08/2026.
//

#include "./reticle.h"

namespace arenai::desktop::gui {

    ReticleDecorator::ReticleDecorator(
        const Rml::NumericValue thickness, const Rml::NumericValue gap)
        : thickness_(thickness), gap_(gap) {}

    Rml::Mesh ReticleDecorator::build_mesh(Rml::Element &element) const {
        const Rml::ColourbPremultiplied color = element_color(element);
        const Rml::Vector2f size = element_size(element);
        const float thickness = element.ResolveLength(thickness_);
        const float gap = element.ResolveLength(gap_);
        const Rml::Vector2f center = size * 0.5f;

        Rml::Mesh mesh;
        if (gap < center.x) {
            append_bar(mesh, {0.f, center.y}, {center.x - gap, center.y}, thickness, color);
            append_bar(mesh, {center.x + gap, center.y}, {size.x, center.y}, thickness, color);
        }
        if (gap < center.y) {
            append_bar(mesh, {center.x, 0.f}, {center.x, center.y - gap}, thickness, color);
            append_bar(mesh, {center.x, center.y + gap}, {center.x, size.y}, thickness, color);
        }
        return mesh;
    }

    ReticleDecoratorInstancer::ReticleDecoratorInstancer()
        : id_thickness_(add_length("thickness", "2dp")), id_gap_(add_length("gap", "6dp")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> ReticleDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<ReticleDecorator>(
            length_of(properties, id_thickness_), length_of(properties, id_gap_));
    }

}// namespace arenai::desktop::gui
