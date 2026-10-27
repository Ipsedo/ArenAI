//
// Created by samuel on 27/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_SLANT_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_SLANT_H

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    // The "Duel néon" control shape: a parallelogram filling the element's
    // border box, its left and right edges leaning by `slant` (top edge
    // starting `slant` to the right, bottom edge ending `slant` short of the
    // right side), with an optional outline of `thickness` drawn just inside
    // the shape. RCSS has no clip-path and the Vulkan backend renders plain
    // geometry only, so the host element keeps a transparent background and
    // border and the decorator draws the whole surface. Stacks before the
    // corner-reticle cursor mark: decorator: slant(...), corner-reticle(...).
    // RCSS: decorator: slant(<fill-color> <line-color> <slant> <thickness>);
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
