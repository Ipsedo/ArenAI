//
// Created by samuel on 23/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_RETICLE_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_RETICLE_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // The aim crosshair: four arms in the element's color stopping `gap`
    // short of the center. RCSS: decorator: reticle(<thickness> <gap>);
    class ReticleDecorator final : public GeometryDecorator {
    public:
        ReticleDecorator(Rml::NumericValue thickness, Rml::NumericValue gap);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::NumericValue thickness_;
        Rml::NumericValue gap_;
    };

    class ReticleDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        ReticleDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_gap_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_RETICLE_H
