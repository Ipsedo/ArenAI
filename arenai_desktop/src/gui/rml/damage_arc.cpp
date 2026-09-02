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

    Rml::DecoratorDataHandle
    DamageArcDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        const auto &computed = element->GetComputedValues();
        const Rml::ColourbPremultiplied color =
            computed.color().ToPremultiplied(computed.opacity());
        const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
        const float thickness = element->ResolveLength(thickness_);

        const Rml::Vector2f center = size * 0.5f;
        const float outer_radius = 0.5f * std::min(size.x, size.y);
        const float inner_radius = std::max(outer_radius - thickness, 0.f);
        const float span = span_degrees_ * (std::numbers::pi_v<float> / 180.f);

        // annular sector centered on the top of the box (the element is
        // rotated from code to point at the impact); the color fades
        // quadratically toward both tips of the arc
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

        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void DamageArcDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void DamageArcDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    DamageArcDecoratorInstancer::DamageArcDecoratorInstancer() {
        id_thickness_ = RegisterProperty("thickness", "16dp").AddParser("length").GetId();
        id_span_ = RegisterProperty("span", "60").AddParser("number").GetId();
        RegisterShorthand("decorator", "thickness, span", Rml::ShorthandType::FallThrough);
    }

    Rml::SharedPtr<Rml::Decorator> DamageArcDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<DamageArcDecorator>(
            properties.GetProperty(id_thickness_)->GetNumericValue(),
            properties.GetProperty(id_span_)->Get<float>());
    }

}// namespace arenai::desktop::gui
