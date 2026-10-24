//
// Created by samuel on 02/08/2026.
//

#include "./corner_reticle.h"

namespace arenai::desktop::gui {

    CornerReticleDecorator::CornerReticleDecorator(
        const Rml::Colourb tick_color, const Rml::Colourb fill_color,
        const Rml::NumericValue tick_length, const Rml::NumericValue thickness,
        const Rml::NumericValue inset, const Rml::NumericValue fill_inset)
        : tick_color_(tick_color), fill_color_(fill_color), tick_length_(tick_length),
          thickness_(thickness), inset_(inset), fill_inset_(fill_inset) {}

    Rml::Mesh CornerReticleDecorator::build_mesh(Rml::Element &element) const {
        const float opacity = element.GetComputedValues().opacity();
        const Rml::Vector2f size = element_size(element);
        const float length = element.ResolveLength(tick_length_);
        const float thickness = element.ResolveLength(thickness_);
        const float inset = element.ResolveLength(inset_);
        const Rml::ColourbPremultiplied tick = tick_color_.ToPremultiplied(opacity);

        Rml::Mesh mesh;

        // each corner is an L: the horizontal tick, then the vertical
        // remainder below/above it (no overlap, so translucent tick colors
        // blend once)
        for (const float sx: {1.f, -1.f})
            for (const float sy: {1.f, -1.f}) {
                const float hx = sx > 0.f ? inset : size.x - inset - length;
                const float hy = sy > 0.f ? inset : size.y - inset - thickness;
                Rml::MeshUtilities::GenerateQuad(mesh, {hx, hy}, {length, thickness}, tick);
                if (length > thickness) {
                    const float vx = sx > 0.f ? inset : size.x - inset - thickness;
                    const float vy = sy > 0.f ? inset + thickness : size.y - inset - length;
                    Rml::MeshUtilities::GenerateQuad(
                        mesh, {vx, vy}, {thickness, length - thickness}, tick);
                }
            }

        // centered fill (slider knob): inset from the edges, the element's
        // own border-radius, skipped when left transparent
        if (fill_color_.alpha > 0) {
            const float fill_inset = element.ResolveLength(fill_inset_);
            const float radius = element.GetComputedValues().border_top_left_radius();
            const Rml::RenderBox fill_box(
                {size.x - 2.f * fill_inset, size.y - 2.f * fill_inset}, {fill_inset, fill_inset},
                {0.f, 0.f, 0.f, 0.f}, {radius, radius, radius, radius});
            Rml::MeshUtilities::GenerateBackground(
                mesh, fill_box, fill_color_.ToPremultiplied(opacity));
        }

        return mesh;
    }

    CornerReticleDecoratorInstancer::CornerReticleDecoratorInstancer()
        : id_tick_color_(add_color("tick-color", "#00A6FB")),
          id_fill_color_(add_color("fill-color", "transparent")),
          id_tick_length_(add_length("tick-length", "7dp")),
          id_thickness_(add_length("thickness", "2dp")), id_inset_(add_length("inset", "3dp")),
          id_fill_inset_(add_length("fill-inset", "0dp")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> CornerReticleDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<CornerReticleDecorator>(
            color_of(properties, id_tick_color_), color_of(properties, id_fill_color_),
            length_of(properties, id_tick_length_), length_of(properties, id_thickness_),
            length_of(properties, id_inset_), length_of(properties, id_fill_inset_));
    }

}// namespace arenai::desktop::gui
