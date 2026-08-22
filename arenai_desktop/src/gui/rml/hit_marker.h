//
// Created by samuel on 23/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_HIT_MARKER_H
#define ARENAI_DESKTOP_GUI_RML_HIT_MARKER_H

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    class HitMarkerDecorator final : public Rml::Decorator {
    public:
        HitMarkerDecorator(Rml::NumericValue thickness, Rml::NumericValue gap);

        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea) const override;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const override;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const override;

    private:
        Rml::NumericValue thickness_;
        Rml::NumericValue gap_;
    };

    class HitMarkerDecoratorInstancer final : public Rml::DecoratorInstancer {
    public:
        HitMarkerDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_thickness_{};
        Rml::PropertyId id_gap_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_HIT_MARKER_H
