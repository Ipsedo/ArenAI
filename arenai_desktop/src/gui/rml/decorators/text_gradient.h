#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_TEXT_GRADIENT_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_TEXT_GRADIENT_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class TextGradientDecorator final : public Rml::Decorator {
    public:
        TextGradientDecorator(
            Rml::Colourb left_color, Rml::Colourb right_color, float start_percent,
            float end_percent);

        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea area) const override;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const override;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const override;

    private:
        Rml::Colourb left_color_;
        Rml::Colourb right_color_;
        float start_percent_;
        float end_percent_;
    };

    class TextGradientDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        TextGradientDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_left_color_;
        Rml::PropertyId id_right_color_;
        Rml::PropertyId id_start_;
        Rml::PropertyId id_end_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_TEXT_GRADIENT_H
