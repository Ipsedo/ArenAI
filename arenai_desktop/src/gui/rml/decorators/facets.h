#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H

#include <vector>

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class FacetsDecorator final : public GeometryDecorator {
    public:
        struct Polygon {
            std::vector<Rml::Vector2f> points;
            float opacity = 1.f;
        };

        FacetsDecorator(
            Rml::Colourb fill_color, Rml::Colourb line_color, Rml::NumericValue thickness,
            const Rml::String &polygons, const Rml::String &animation);

        static std::vector<Polygon> parse(const Rml::String &polygons);

        static float swell_offset(Rml::Vector2f point, double time);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;
        bool animated() const override;

    private:
        Rml::Colourb fill_color_;
        Rml::Colourb line_color_;
        Rml::NumericValue thickness_;
        std::vector<Polygon> polygons_;
        bool houle_;
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
        Rml::PropertyId id_animation_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_FACETS_H
