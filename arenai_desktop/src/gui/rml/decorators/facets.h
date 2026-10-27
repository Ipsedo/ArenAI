//
// Created by samuel on 27/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H

#include <vector>

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // The low-poly backdrop of the "Duel néon" screens: a list of flat
    // polygons (the arena floor's facets, the horizon shards, the center
    // line's dashes) drawn in one color, each with its own opacity, filled
    // and/or outlined. Coordinates are authored on the 1280×720 design
    // canvas of the mockups and stretched to the element's border box, so
    // one declaration covers every window size.
    // RCSS: decorator: facets(<fill-color> <line-color> <thickness> "<polygons>");
    //   polygons: `x y x y x y[@opacity]` entries separated by `|`; a
    //   two-point entry is a segment of `thickness`; a transparent color
    //   skips that pass (fill or outline).
    class FacetsDecorator final : public GeometryDecorator {
    public:
        struct Polygon {
            std::vector<Rml::Vector2f> points;
            float opacity = 1.f;
        };

        FacetsDecorator(
            Rml::Colourb fill_color, Rml::Colourb line_color, Rml::NumericValue thickness,
            const Rml::String &polygons);

        // the parsed form of the RCSS string (exposed for the unit tests)
        static std::vector<Polygon> parse(const Rml::String &polygons);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::Colourb fill_color_;
        Rml::Colourb line_color_;
        Rml::NumericValue thickness_;
        std::vector<Polygon> polygons_;
    };

    class FacetsDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        FacetsDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_fill_color_;
        Rml::PropertyId id_line_color_;
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_polygons_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H
