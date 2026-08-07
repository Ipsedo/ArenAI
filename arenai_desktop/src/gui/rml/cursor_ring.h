//
// Created by samuel on 02/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_CURSOR_RING_H
#define ARENAI_DESKTOP_GUI_RML_CURSOR_RING_H

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // The slider knob is a widget-internal element (sliderbar) that cannot
    // be wrapped in RML like the buttons, so its detached cursor ring is a
    // gui-local decorator instead: an ink ring hugging the element's edge,
    // a transparent gap, and the fill disc — plain triangle geometry, the
    // only thing the Vulkan backend renders (no box-shadow, no textures).
    // RCSS: decorator: cursor-ring(<ring-color> <fill-color> <ring-width> <gap>);
    class CursorRingDecorator final : public Rml::Decorator {
    public:
        CursorRingDecorator(
            Rml::Colourb ring_color, Rml::Colourb fill_color, Rml::NumericValue ring_width,
            Rml::NumericValue gap);

        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea) const override;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const override;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const override;

    private:
        Rml::Colourb ring_color_;
        Rml::Colourb fill_color_;
        Rml::NumericValue ring_width_;
        Rml::NumericValue gap_;
    };

    class CursorRingDecoratorInstancer final : public Rml::DecoratorInstancer {
    public:
        CursorRingDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_ring_color_{};
        Rml::PropertyId id_fill_color_{};
        Rml::PropertyId id_ring_width_{};
        Rml::PropertyId id_gap_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_CURSOR_RING_H
