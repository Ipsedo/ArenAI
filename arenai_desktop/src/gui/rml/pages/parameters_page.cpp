//
// Created by samuel on 25/10/2026.
//

#include "./parameters_page.h"

#include "../labels.h"

namespace arenai::desktop::gui {

    ParametersPage::ParametersPage(MenuState &state)
        : MenuPage(PageId::Parameters, PageId::Main, "menu/parameters.rml", "parameters"),
          state_(state) {}

    void ParametersPage::bind_controls(DataModel &model) {
        model.bind("nb_tanks", &state_.settings.nb_tanks);
        model.bind("spawn_side", &state_.settings.spawn_side);
        model.bind("agent_valid", &state_.agent_valid);
        model.bind("graphics_summary", &graphics_summary_);
        model.bind("controls_summary", &controls_summary_);
        model.bind("ai_summary", &ai_summary_);
        refresh_summaries();
    }

    void ParametersPage::on_show() {
        refresh_summaries();
        model().dirty({"graphics_summary", "controls_summary", "ai_summary"});
    }

    void ParametersPage::refresh_summaries() {
        graphics_summary_ = graphics_summary(state_.settings);
        controls_summary_ = controller_label(state_.settings.controller_kind);
        ai_summary_ = agent_summary(state_.settings, state_.agent_valid);
    }

}// namespace arenai::desktop::gui
