//
// Created by samuel on 02/08/2026.
//

#include "./cursor_ring.h"

namespace arenai::desktop::gui {

    CursorRingDecorator::CursorRingDecorator(
        const Rml::Colourb ring_color, const Rml::Colourb fill_color,
        const Rml::NumericValue ring_width, const Rml::NumericValue gap)
        : ring_color_(ring_color), fill_color_(fill_color), ring_width_(ring_width), gap_(gap) {}

    Rml::DecoratorDataHandle
    CursorRingDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        const float opacity = element->GetComputedValues().opacity();
        const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
        const float ring_width = element->ResolveLength(ring_width_);
        const float inset = ring_width + element->ResolveLength(gap_);
        // ring and fill both adopt the element's border-radius: the
        // ring must read as the same shape as the knob it surrounds
        // (a concentric radius+inset ring looks round next to a
        // square knob), so no radius inflation on the outer box
        const float ring_radius = element->GetComputedValues().border_top_left_radius();
        const float fill_radius = ring_radius;

        Rml::Mesh mesh;

        // hollow ring at the outer edge (zero-alpha background: only
        // the border geometry is emitted)
        const Rml::RenderBox ring_box(
            {size.x - 2.f * ring_width, size.y - 2.f * ring_width}, {0.f, 0.f},
            {ring_width, ring_width, ring_width, ring_width},
            {ring_radius, ring_radius, ring_radius, ring_radius});
        const Rml::ColourbPremultiplied ring = ring_color_.ToPremultiplied(opacity);
        const Rml::ColourbPremultiplied ring_colors[4] = {ring, ring, ring, ring};
        Rml::MeshUtilities::GenerateBackgroundBorder(
            mesh, ring_box, Rml::ColourbPremultiplied(0, 0, 0, 0), ring_colors);

        // fill disc, inset past the ring and the gap
        const Rml::Vector2f fill_size = {size.x - 2.f * inset, size.y - 2.f * inset};
        const Rml::RenderBox fill_box(
            fill_size, {inset, inset}, {0.f, 0.f, 0.f, 0.f},
            {fill_radius, fill_radius, fill_radius, fill_radius});
        Rml::MeshUtilities::GenerateBackground(
            mesh, fill_box, fill_color_.ToPremultiplied(opacity));

        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void
    CursorRingDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void CursorRingDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    CursorRingDecoratorInstancer::CursorRingDecoratorInstancer() {
        id_ring_color_ = RegisterProperty("ring-color", "#EAF7FF").AddParser("color").GetId();
        id_fill_color_ = RegisterProperty("fill-color", "#00A6FB").AddParser("color").GetId();
        id_ring_width_ = RegisterProperty("ring-width", "2dp").AddParser("length").GetId();
        id_gap_ = RegisterProperty("gap", "2dp").AddParser("length").GetId();
        RegisterShorthand(
            "decorator", "ring-color, fill-color, ring-width, gap",
            Rml::ShorthandType::FallThrough);
    }

    Rml::SharedPtr<Rml::Decorator> CursorRingDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<CursorRingDecorator>(
            properties.GetProperty(id_ring_color_)->Get<Rml::Colourb>(),
            properties.GetProperty(id_fill_color_)->Get<Rml::Colourb>(),
            properties.GetProperty(id_ring_width_)->GetNumericValue(),
            properties.GetProperty(id_gap_)->GetNumericValue());
    }

}// namespace arenai::desktop::gui
