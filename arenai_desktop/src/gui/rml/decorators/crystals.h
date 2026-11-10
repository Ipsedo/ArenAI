#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H

#include <vector>

#include "./geometry_decorator.h"

namespace arenai::desktop::gui {

    class CrystalsDecorator final : public GeometryDecorator {
    public:
        struct Instance {
            Rml::Vector2f center;
            float size = 0.f;
            int shape = 1;
            float spin = 1.f;
            float phase = 0.f;
        };

        CrystalsDecorator(
            Rml::Colourb line_color, Rml::NumericValue thickness, const Rml::String &instances);

        static std::vector<Instance> parse(const Rml::String &instances);

    protected:
        Rml::Mesh build_mesh(Rml::Element &element) const override;
        bool animated() const override { return true; }

    private:
        Rml::Colourb line_color_;
        Rml::NumericValue thickness_;
        std::vector<Instance> instances_;
    };

    class CrystalsDecoratorInstancer final : public GeometryDecoratorInstancer {
    public:
        CrystalsDecoratorInstancer();

        Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
            const Rml::String &, const Rml::PropertyDictionary &properties,
            const Rml::DecoratorInstancerInterface &) override;

    private:
        Rml::PropertyId id_line_color_;
        Rml::PropertyId id_thickness_;
        Rml::PropertyId id_instances_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_CRYSTALS_H
