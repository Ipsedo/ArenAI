//
// Created by samuel on 27/10/2026.
//

#include "./click_outside_listener.h"

#include <utility>

namespace arenai::desktop::gui {

    ClickOutsideListener::ClickOutsideListener(std::function<void()> on_outside)
        : on_outside_(std::move(on_outside)) {}

    void ClickOutsideListener::ProcessEvent(Rml::Event &event) {
        if (!inside_dropdown(event.GetTargetElement())) on_outside_();
    }

    bool inside_dropdown(const Rml::Element *element) {
        for (; element != nullptr; element = element->GetParentNode())
            if (element->IsClassSet("dropdown")) return true;
        return false;
    }

}// namespace arenai::desktop::gui
