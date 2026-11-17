#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_VIEW_CONE_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_VIEW_CONE_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // a fan pointing up from the box center, the element's color fading to the rim
    class ViewConeDecorator final : public GeometryDecorator {
    public:
        explicit ViewConeDecorator(float span_degrees);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;

    private:
        float span_degrees_;
    };

    class ViewConeDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        ViewConeDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_span_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_VIEW_CONE_H
