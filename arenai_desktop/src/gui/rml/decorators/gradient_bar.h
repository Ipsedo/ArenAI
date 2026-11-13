#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_GRADIENT_BAR_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_GRADIENT_BAR_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class GradientBarDecorator final : public GeometryDecorator {
    public:
        GradientBarDecorator(Rml::Colourb left_color, Rml::Colourb right_color, float fade_percent);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::Colourb left_color_;
        Rml::Colourb right_color_;
        float fade_percent_;
    };

    class GradientBarDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        GradientBarDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_left_color_;
        Rml::PropertyId id_right_color_;
        Rml::PropertyId id_fade_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_GRADIENT_BAR_H
