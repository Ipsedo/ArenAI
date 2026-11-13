#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_SLANT_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_SLANT_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class SlantDecorator final : public GeometryDecorator {
    public:
        SlantDecorator(
            Rml::Colourb fill_color, Rml::Colourb line_color, Rml::NumericValue slant,
            Rml::NumericValue thickness);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::Colourb fill_color_;
        Rml::Colourb line_color_;
        Rml::NumericValue slant_;
        Rml::NumericValue thickness_;
    };

    class SlantDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        SlantDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_fill_color_;
        Rml::PropertyId id_line_color_;
        Rml::PropertyId id_slant_;
        Rml::PropertyId id_thickness_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_SLANT_H
