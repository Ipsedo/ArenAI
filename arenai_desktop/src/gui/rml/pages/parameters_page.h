//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_PARAMETERS_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_PARAMETERS_PAGE_H

#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // The Parameters hub: the game sliders plus the doors to the Graphics,
    // Controls and AI screens.
    class ParametersPage final : public MenuPage {
    public:
        explicit ParametersPage(MenuState &state);

    protected:
        void bind_controls(DataModel &model) override;

    private:
        MenuState &state_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_PARAMETERS_PAGE_H
