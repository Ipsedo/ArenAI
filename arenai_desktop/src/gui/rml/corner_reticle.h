//
// Created by samuel on 02/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_CORNER_RETICLE_H
#define ARENAI_DESKTOP_GUI_RML_CORNER_RETICLE_H

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // The menu's single cursor mark: four L-shaped corner ticks framing the
    // control under the cursor (mouse hover or gamepad focus). The optional
    // centered fill exists for the slider knob, a widget-internal element
    // (sliderbar) that cannot host children — plain triangle geometry, the
    // only thing the Vulkan backend renders (no box-shadow, no textures).
    // RCSS: decorator: corner-reticle(
    //           <tick-color> <fill-color> <tick-length> <thickness> <inset> <fill-inset>);
    class CornerReticleDecorator final : public Rml::Decorator {
    public:
        CornerReticleDecorator(
            Rml::Colourb tick_color, Rml::Colourb fill_color, Rml::NumericValue tick_length,
            Rml::NumericValue thickness, Rml::NumericValue inset, Rml::NumericValue fill_inset);

        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea) const override;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const override;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const override;

    private:
        Rml::Colourb tick_color_;
        Rml::Colourb fill_color_;
        Rml::NumericValue tick_length_;
        Rml::NumericValue thickness_;
        Rml::NumericValue inset_;
        Rml::NumericValue fill_inset_;
    };

    class CornerReticleDecoratorInstancer final : public Rml::DecoratorInstancer {
    public:
        CornerReticleDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_tick_color_{};
        Rml::PropertyId id_fill_color_{};
        Rml::PropertyId id_tick_length_{};
        Rml::PropertyId id_thickness_{};
        Rml::PropertyId id_inset_{};
        Rml::PropertyId id_fill_inset_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_CORNER_RETICLE_H
