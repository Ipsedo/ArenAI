#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_CORNER_RETICLE_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_CORNER_RETICLE_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class CornerReticleDecorator final : public GeometryDecorator {
    public:
        CornerReticleDecorator(
            Rml::Colourb tick_color, Rml::Colourb fill_color, Rml::NumericValue tick_length,
            Rml::NumericValue thickness, Rml::NumericValue inset, Rml::NumericValue fill_inset);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::Colourb tick_color_;
        Rml::Colourb fill_color_;
        Rml::NumericValue tick_length_;
        Rml::NumericValue thickness_;
        Rml::NumericValue inset_;
        Rml::NumericValue fill_inset_;
    };

    class CornerReticleDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        CornerReticleDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_tick_color_;
        Rml::PropertyId id_fill_color_;
        Rml::PropertyId id_tick_length_;
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_inset_;
        Rml::PropertyId id_fill_inset_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_CORNER_RETICLE_H
