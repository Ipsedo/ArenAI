//
// Created by samuel on 23/08/2026.
//

#include "./reticle.h"

#include <algorithm>

namespace arenai::desktop::gui {

    namespace {

        void append_bar(
            Rml::Mesh &mesh, const Rml::Vector2f from, const Rml::Vector2f to,
            const float thickness, const Rml::ColourbPremultiplied color) {
            const Rml::Vector2f min(
                std::min(from.x, to.x) - (from.x == to.x ? 0.5f * thickness : 0.f),
                std::min(from.y, to.y) - (from.y == to.y ? 0.5f * thickness : 0.f));
            const Rml::Vector2f max(
                std::max(from.x, to.x) + (from.x == to.x ? 0.5f * thickness : 0.f),
                std::max(from.y, to.y) + (from.y == to.y ? 0.5f * thickness : 0.f));
            Rml::MeshUtilities::GenerateQuad(mesh, min, max - min, color);
        }

    }// namespace

    ReticleDecorator::ReticleDecorator(
        const Rml::NumericValue thickness, const Rml::NumericValue gap)
        : thickness_(thickness), gap_(gap) {}

    Rml::DecoratorDataHandle
    ReticleDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        const auto &computed = element->GetComputedValues();
        const Rml::ColourbPremultiplied color =
            computed.color().ToPremultiplied(computed.opacity());
        const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
        const float thickness = element->ResolveLength(thickness_);
        const float gap = element->ResolveLength(gap_);

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

        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void ReticleDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void ReticleDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    ReticleDecoratorInstancer::ReticleDecoratorInstancer() {
        id_thickness_ = RegisterProperty("thickness", "2dp").AddParser("length").GetId();
        id_gap_ = RegisterProperty("gap", "6dp").AddParser("length").GetId();
        RegisterShorthand("decorator", "thickness, gap", Rml::ShorthandType::FallThrough);
    }

    Rml::SharedPtr<Rml::Decorator> ReticleDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<ReticleDecorator>(
            properties.GetProperty(id_thickness_)->GetNumericValue(),
            properties.GetProperty(id_gap_)->GetNumericValue());
    }

}// namespace arenai::desktop::gui
