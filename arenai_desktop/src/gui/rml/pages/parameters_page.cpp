//
// Created by samuel on 25/10/2026.
//

#include "./parameters_page.h"

namespace arenai::desktop::gui {

    ParametersPage::ParametersPage(MenuState &state)
        : MenuPage(PageId::Parameters, PageId::Main, "menu/parameters.rml", "parameters"),
          state_(state) {}

    void ParametersPage::bind_controls(DataModel &model) {
        model.bind("nb_tanks", &state_.settings.nb_tanks);
        model.bind("spawn_side", &state_.settings.spawn_side);
        model.bind("agent_valid", &state_.agent_valid);
    }

}// namespace arenai::desktop::gui
