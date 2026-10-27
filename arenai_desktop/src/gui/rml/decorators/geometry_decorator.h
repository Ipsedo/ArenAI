//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_GEOMETRY_DECORATOR_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_GEOMETRY_DECORATOR_H

#include <vector>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // Template Method for the menu's flat-geometry decorators: a subclass
    // only describes the mesh of one element, the base owns the RmlUi
    // element-data lifecycle (compile at generation, release, render at the
    // element's border-box offset).
    class GeometryDecorator : public Rml::Decorator {
    public:
        Rml::DecoratorDataHandle
        GenerateElementData(Rml::Element *element, Rml::BoxArea area) const final;
        void ReleaseElementData(Rml::DecoratorDataHandle element_data) const final;
        void
        RenderElement(Rml::Element *element, Rml::DecoratorDataHandle element_data) const final;

    protected:
        virtual Rml::Mesh build_mesh(Rml::Element &element) const = 0;

        // the element's `color` premultiplied by its opacity
        static Rml::ColourbPremultiplied element_color(const Rml::Element &element);
        // the element's border-box size, the decorator's drawing area
        static Rml::Vector2f element_size(Rml::Element &element);
    };

    // a bar of the given thickness between two points (any orientation)
    void append_bar(
        Rml::Mesh &mesh, Rml::Vector2f from, Rml::Vector2f to, float thickness,
        Rml::ColourbPremultiplied color);

    // Base instancer: declares the decorator's RCSS properties in order and
    // exposes them as the positional `decorator: name(a b c)` shorthand.
    class GeometryDecoratorInstancer : public Rml::DecoratorInstancer {
    protected:
        Rml::PropertyId add_length(const char *name, const char *default_value);
        Rml::PropertyId add_color(const char *name, const char *default_value);
        Rml::PropertyId add_number(const char *name, const char *default_value);
        // a quoted free-form value ("..." in the RCSS shorthand)
        Rml::PropertyId add_string(const char *name, const char *default_value);
        // call once every property is declared
        void register_shorthand();

        static Rml::NumericValue
        length_of(const Rml::PropertyDictionary &properties, Rml::PropertyId id);
        static Rml::Colourb color_of(const Rml::PropertyDictionary &properties, Rml::PropertyId id);
        static float number_of(const Rml::PropertyDictionary &properties, Rml::PropertyId id);
        static Rml::String string_of(const Rml::PropertyDictionary &properties, Rml::PropertyId id);

    private:
        Rml::PropertyId add(const char *name, const char *default_value, const char *parser);

        std::vector<Rml::String> property_names_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_GEOMETRY_DECORATOR_H
