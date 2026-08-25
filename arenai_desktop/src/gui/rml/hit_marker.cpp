//
// Created by samuel on 23/08/2026.
//

#include "./hit_marker.h"

#include <cmath>

namespace arenai::desktop::gui {

    namespace {

        void append_bar(
            Rml::Mesh &mesh, const Rml::Vector2f from, const Rml::Vector2f to,
            const float thickness, const Rml::ColourbPremultiplied color) {
            const Rml::Vector2f direction = to - from;
            const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
            if (length <= 0.f) return;
            const Rml::Vector2f normal =
                Rml::Vector2f(-direction.y, direction.x) * (0.5f * thickness / length);

            const int base = static_cast<int>(mesh.vertices.size());
            for (const Rml::Vector2f position:
                 {from + normal, to + normal, to - normal, from - normal}) {
                Rml::Vertex vertex;
                vertex.position = position;
                vertex.colour = color;
                vertex.tex_coord = {0.f, 0.f};
                mesh.vertices.push_back(vertex);
            }
            for (const int index: {0, 1, 2, 0, 2, 3}) mesh.indices.push_back(base + index);
        }

    }// namespace

    HitMarkerDecorator::HitMarkerDecorator(
        const Rml::NumericValue thickness, const Rml::NumericValue gap)
        : thickness_(thickness), gap_(gap) {}

    Rml::DecoratorDataHandle
    HitMarkerDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        const auto &computed = element->GetComputedValues();
        const Rml::ColourbPremultiplied color =
            computed.color().ToPremultiplied(computed.opacity());
        const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
        const float thickness = element->ResolveLength(thickness_);
        const float gap = element->ResolveLength(gap_);

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

        auto *geometry =
            new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
        return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
    }

    void HitMarkerDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<Rml::Geometry *>(element_data);
    }

    void HitMarkerDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        reinterpret_cast<Rml::Geometry *>(element_data)
            ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
    }

    HitMarkerDecoratorInstancer::HitMarkerDecoratorInstancer() {
        id_thickness_ = RegisterProperty("thickness", "3dp").AddParser("length").GetId();
        id_gap_ = RegisterProperty("gap", "6dp").AddParser("length").GetId();
        RegisterShorthand("decorator", "thickness, gap", Rml::ShorthandType::FallThrough);
    }

    Rml::SharedPtr<Rml::Decorator> HitMarkerDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<HitMarkerDecorator>(
            properties.GetProperty(id_thickness_)->GetNumericValue(),
            properties.GetProperty(id_gap_)->GetNumericValue());
    }

}// namespace arenai::desktop::gui
