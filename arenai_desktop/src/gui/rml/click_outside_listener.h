//
// Created by samuel on 27/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H
#define ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H

#include <functional>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // Document-level click listener folding a page's dropdowns: fires the
    // callback for any click landing outside an element of class
    // `.dropdown` (the head and its list both live inside one).
    class ClickOutsideListener final : public Rml::EventListener {
    public:
        explicit ClickOutsideListener(std::function<void()> on_outside);
        void ProcessEvent(Rml::Event &event) override;

    private:
        std::function<void()> on_outside_;
    };

    // true when `element` or one of its ancestors carries the .dropdown class
    bool inside_dropdown(const Rml::Element *element);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H
