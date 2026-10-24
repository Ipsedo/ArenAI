//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DECORATORS_DECORATOR_REGISTRY_H
#define ARENAI_DESKTOP_GUI_RML_DECORATORS_DECORATOR_REGISTRY_H

#include <memory>
#include <vector>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // Owns the decorator instancers registered with RmlUi. Built after
    // Rml::Initialise() (property registration needs the style-sheet
    // specification) and, as RmlUi requires, destroyed after Rml::Shutdown().
    class DecoratorRegistry {
    public:
        DecoratorRegistry() = default;
        DecoratorRegistry(const DecoratorRegistry &) = delete;
        DecoratorRegistry &operator=(const DecoratorRegistry &) = delete;

        void add(const Rml::String &name, std::unique_ptr<Rml::DecoratorInstancer> instancer);

    private:
        std::vector<std::unique_ptr<Rml::DecoratorInstancer>> instancers_;
    };

    // the decorators menu.rcss and hud.rcss rely on
    void register_menu_decorators(DecoratorRegistry &registry);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DECORATORS_DECORATOR_REGISTRY_H
