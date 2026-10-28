//
// Created by samuel on 28/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H

#include <vector>

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // The "Duel néon" horizon crystals: irregular wireframe tetrahedra
    // tumbling in 3D above the arena, one of two authored shapes per
    // instance, projected with a light perspective (back edges dimmed) and a
    // slow vertical bob. Coordinates live on the same 1280×720 design canvas
    // as facets; the mesh is rebuilt every frame.
    // RCSS: decorator: crystals(<line-color> <thickness> "<instances>");
    //   instances: `cx cy size shape spin phase` entries separated by `|`;
    //   shape picks the authored tetrahedron (1 or 2), spin is a signed
    //   rotation-speed factor, phase offsets the rotation and the bob.
    class CrystalsDecorator final : public GeometryDecorator {
    public:
        struct Instance {
            Rml::Vector2f center;
            float size = 0.f;
            int shape = 1;
            float spin = 1.f;
            float phase = 0.f;
        };

        CrystalsDecorator(
            Rml::Colourb line_color, Rml::NumericValue thickness, const Rml::String &instances);

        // the parsed form of the RCSS string (exposed for the unit tests)
        static std::vector<Instance> parse(const Rml::String &instances);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;
        bool animated() const override { return true; }

    private:
        Rml::Colourb line_color_;
        Rml::NumericValue thickness_;
        std::vector<Instance> instances_;
    };

    class CrystalsDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        CrystalsDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_line_color_;
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_instances_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H
