//
// Created by samuel on 27/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DAMAGE_ARC_H
#define ARENAI_DESKTOP_GUI_RML_DAMAGE_ARC_H

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    class DamageArcDecorator final : public Rml::Decorator {
    public:
        DamageArcDecorator(Rml::NumericValue thickness, float span_degrees);

        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea) const override;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const override;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const override;

    private:
        Rml::NumericValue thickness_;
        float span_degrees_;
    };

    class DamageArcDecoratorInstancer final : public Rml::DecoratorInstancer {
    public:
        DamageArcDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_thickness_{};
        Rml::PropertyId id_span_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DAMAGE_ARC_H
