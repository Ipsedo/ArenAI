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

    Rml::DecoratorDataHandle
    CornerReticleDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        const float opacity = element->GetComputedValues().opacity();
        const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
        const float length = element->ResolveLength(tick_length_);
        const float thickness = element->ResolveLength(thickness_);
        const float inset = element->ResolveLength(inset_);
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
            const float fill_inset = element->ResolveLength(fill_inset_);
            const float radius = element->GetComputedValues().border_top_left_radius();
            const Rml::RenderBox fill_box(
                {size.x - 2.f * fill_inset, size.y - 2.f * fill_inset}, {fill_inset, fill_inset},
                {0.f, 0.f, 0.f, 0.f}, {radius, radius, radius, radius});
            Rml::MeshUtilities::GenerateBackground(
                mesh, fill_box, fill_color_.ToPremultiplied(opacity));
        }

        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void
    CornerReticleDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void CornerReticleDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    CornerReticleDecoratorInstancer::CornerReticleDecoratorInstancer() {
        id_tick_color_ = RegisterProperty("tick-color", "#00A6FB").AddParser("color").GetId();
        id_fill_color_ = RegisterProperty("fill-color", "transparent").AddParser("color").GetId();
        id_tick_length_ = RegisterProperty("tick-length", "7dp").AddParser("length").GetId();
        id_thickness_ = RegisterProperty("thickness", "2dp").AddParser("length").GetId();
        id_inset_ = RegisterProperty("inset", "3dp").AddParser("length").GetId();
        id_fill_inset_ = RegisterProperty("fill-inset", "0dp").AddParser("length").GetId();
        RegisterShorthand(
            "decorator", "tick-color, fill-color, tick-length, thickness, inset, fill-inset",
            Rml::ShorthandType::FallThrough);
    }

    Rml::SharedPtr<Rml::Decorator> CornerReticleDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<CornerReticleDecorator>(
            properties.GetProperty(id_tick_color_)->Get<Rml::Colourb>(),
            properties.GetProperty(id_fill_color_)->Get<Rml::Colourb>(),
            properties.GetProperty(id_tick_length_)->GetNumericValue(),
            properties.GetProperty(id_thickness_)->GetNumericValue(),
            properties.GetProperty(id_inset_)->GetNumericValue(),
            properties.GetProperty(id_fill_inset_)->GetNumericValue());
    }

}// namespace arenai::desktop::gui
