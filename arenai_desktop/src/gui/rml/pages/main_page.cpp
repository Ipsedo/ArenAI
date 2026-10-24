//
// Created by samuel on 25/10/2026.
//

#include "./main_page.h"

#include <utility>

namespace arenai::desktop::gui {

    MainPage::MainPage(MenuState &state, std::function<void(MenuOutcome)> on_outcome)
        : MenuPage(PageId::Main, std::nullopt, "menu/main_menu.rml", "main"), state_(state),
          on_outcome_(std::move(on_outcome)) {}

    void MainPage::bind_controls(DataModel &model) {
        model.bind("can_play", &state_.agent_valid);

        model.on_event("play", [this](const Rml::VariantList &) {
            if (state_.agent_valid) on_outcome_(MenuOutcome::Play);
        });
        model.on_event(
            "exit", [this](const Rml::VariantList &) { on_outcome_(MenuOutcome::Quit); });
    }

}// namespace arenai::desktop::gui
