#include "./popup_page.h"

#include <cmath>
#include <utility>

#include "./labels.h"

namespace arenai::desktop::gui {

    namespace {

        constexpr const char *document_path(const PopupKind kind) {
            switch (kind) {
                case PopupKind::Pause: return "menu/pause.rml";
                case PopupKind::Victory: return "menu/victory.rml";
                default: return "menu/game_over.rml";
            }
        }

        constexpr const char *model_name(const PopupKind kind) {
            switch (kind) {
                case PopupKind::Pause: return "pause";
                case PopupKind::Victory: return "victory";
                default: return "game_over";
            }
        }

    }// namespace

    PopupPage::PopupPage(const PopupKind kind, MenuState &state)
        : DocumentView(document_path(kind), model_name(kind)), kind_(kind), state_(state) {}

    void PopupPage::bind(DataModel &model) {
        model.bind("score", &state_.match.player_score);
        model.bind("machine_score", &machine_score_);
        model.bind("enemies_alive", &state_.match.enemies_alive);
        model.bind("clock", &clock_);

        model.on_event(
            "popup_continue", [this](const Rml::VariantList &) { report(PopupAction::Continue); });
        model.on_event(
            "popup_main_menu", [this](const Rml::VariantList &) { report(PopupAction::MainMenu); });
        model.on_event(
            "popup_exit", [this](const Rml::VariantList &) { report(PopupAction::ExitGame); });
        model.on_event(
            "popup_retry", [this](const Rml::VariantList &) { report(PopupAction::Retry); });
    }

    void PopupPage::open(const MatchStatus &status, PopupListener on_action) {
        state_.match = status;
        clock_ = clock_label(status.elapsed_seconds);
        machine_score_ = status.enemies_max_score;
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
        if (const auto listener = on_action_) listener(action);
    }

}// namespace arenai::desktop::gui
