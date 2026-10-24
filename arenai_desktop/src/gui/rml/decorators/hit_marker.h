//
// Created by samuel on 23/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_HIT_MARKER_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_HIT_MARKER_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // The hit marker X: four diagonal ticks from the corners stopping `gap`
    // short of the center. RCSS: decorator: hit-marker(<thickness> <gap>);
    class HitMarkerDecorator final : public GeometryDecorator {
    public:
        HitMarkerDecorator(Rml::NumericValue thickness, Rml::NumericValue gap);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::NumericValue thickness_;
        Rml::NumericValue gap_;
    };

    class HitMarkerDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        HitMarkerDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_gap_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_HIT_MARKER_H
