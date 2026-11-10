#ifndef ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H
#define ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H

#include <functional>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    class ClickOutsideListener final : public Rml::EventListener {
    public:
        explicit ClickOutsideListener(std::function<void()> on_outside);
        void ProcessEvent(Rml::Event &event) override;

    private:
        std::function<void()> on_outside_;
    };

    bool inside_dropdown(const Rml::Element *element);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_CLICK_OUTSIDE_LISTENER_H
