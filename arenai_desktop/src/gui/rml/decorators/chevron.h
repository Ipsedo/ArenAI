#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_CHEVRON_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_CHEVRON_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // an arrowhead pointing up, filled with the element's color over an outline
    class ChevronDecorator final : public GeometryDecorator {
    public:
        ChevronDecorator(Rml::Colourb outline_color, Rml::NumericValue outline_width);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        Rml::Colourb outline_color_;
        Rml::NumericValue outline_width_;
    };

    class ChevronDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        ChevronDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_outline_color_;
        Rml::PropertyId id_outline_width_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_CHEVRON_H
