//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_POPUP_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_POPUP_PAGE_H

#include "../gui.h"
#include "./document_view.h"
#include "./menu_state.h"

namespace arenai::desktop::gui {

    // One in-game popup (pause or game over): shows the match status and reports
    // the player's choice to the listener given at open time.
    class PopupPage final : public DocumentView {
    public:
        PopupPage(PopupKind kind, MenuState &state);

        PopupKind kind() const { return kind_; }

        void open(const MatchStatus &status, PopupListener on_action);
        void close();

        // Escape / gamepad B: resumes a paused game; the game-over popup
        // cannot be backed out of
        void on_escape();

    protected:
        void bind(DataModel &model) override;

    private:
        void report(PopupAction action);

        PopupKind kind_;
        MenuState &state_;
        // the status in the shapes the RML binds
        Rml::String clock_;
        int machine_score_ = 0;
        PopupListener on_action_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_POPUP_PAGE_H
