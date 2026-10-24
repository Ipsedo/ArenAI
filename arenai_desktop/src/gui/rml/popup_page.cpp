//
// Created by samuel on 25/10/2026.
//

#include "./popup_page.h"

#include <utility>

namespace arenai::desktop::gui {

    namespace {

        constexpr const char *document_path(const PopupKind kind) {
            return kind == PopupKind::Pause ? "menu/pause.rml" : "menu/game_over.rml";
        }

        constexpr const char *model_name(const PopupKind kind) {
            return kind == PopupKind::Pause ? "pause" : "game_over";
        }

    }// namespace

    PopupPage::PopupPage(const PopupKind kind, MenuState &state)
        : DocumentView(document_path(kind), model_name(kind)), kind_(kind), state_(state) {}

    void PopupPage::bind(DataModel &model) {
        model.bind("score", &state_.score);

        model.on_event(
            "popup_continue", [this](const Rml::VariantList &) { report(PopupAction::Continue); });
        model.on_event(
            "popup_main_menu", [this](const Rml::VariantList &) { report(PopupAction::MainMenu); });
        model.on_event(
            "popup_exit", [this](const Rml::VariantList &) { report(PopupAction::ExitGame); });
        model.on_event(
            "popup_retry", [this](const Rml::VariantList &) { report(PopupAction::Retry); });
    }

    void PopupPage::open(const int score, PopupListener on_action) {
        state_.score = score;
        on_action_ = std::move(on_action);
        show();
    }

    void PopupPage::close() {
        hide();
        on_action_ = nullptr;
    }

    void PopupPage::on_escape() {
        if (kind_ == PopupKind::Pause) report(PopupAction::Continue);
    }

    void PopupPage::report(const PopupAction action) {
        // local copy: the listener may close this popup from inside the call
        if (const auto listener = on_action_) listener(action);
    }

}// namespace arenai::desktop::gui
