//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_MAIN_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_MAIN_PAGE_H

#include <functional>

#include "../../gui.h"
#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // The root screen: Play (gated by the agent check), Parameters, Exit,
    // plus the footer summarizing the input device and the opponent.
    class MainPage final : public MenuPage {
    public:
        MainPage(MenuState &state, std::function<void(MenuOutcome)> on_outcome);

    protected:
        void bind_controls(DataModel &model) override;
        void on_show() override;

    private:
        void refresh_summaries();

        MenuState &state_;
        std::function<void(MenuOutcome)> on_outcome_;
        Rml::String input_summary_;
        Rml::String agent_summary_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_MAIN_PAGE_H
