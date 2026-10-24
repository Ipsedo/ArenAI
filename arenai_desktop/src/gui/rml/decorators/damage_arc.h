//
// Created by samuel on 27/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_DAMAGE_ARC_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_DAMAGE_ARC_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // An annular sector centered on the top of the box, fading toward both
    // tips; code rotates the element toward the impact.
    // RCSS: decorator: damage-arc(<thickness> <span-degrees>);
    class DamageArcDecorator final : public GeometryDecorator {
    public:
        DamageArcDecorator(Rml::NumericValue thickness, float span_degrees);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::NumericValue thickness_;
        float span_degrees_;
    };

    class DamageArcDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        DamageArcDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_span_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_DAMAGE_ARC_H
