//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H

#include <filesystem>
#include <vector>

#include <RmlUi/Core.h>

#include "../../menu.h"
#include "../input.h"

namespace arenai::desktop::gui {

    // The AI screen: the algorithm toggle and the file explorer picking the
    // trained model's folder, dry-run validated through the injected
    // AgentValidator. Registers its slice of the shared "settings" data model
    // and writes straight into the GameSettings it was built around.
    class AiPage {
    public:
        AiPage(GameSettings &settings, AgentValidator validator);

        // registers the page's variables and callbacks into the shared model
        void bind(Rml::DataModelConstructor &constructor);
        void set_model_handle(Rml::DataModelHandle handle);

        // hooks the D-pad bridge across the explorer's scroll container and
        // remembers the document for the post-Update focus restore
        void attach(Rml::ElementDocument &document);

        void refresh_explorer();

        // after the context Update rebuilt the entry clones, put the gamepad
        // cursor back on the first entry of the fresh listing
        void update_after_context();

        // the menu's Play gate: true once the dry-run load succeeded
        bool can_play() const { return can_play_; }

    private:
        void validate_agent();
        void focus_first_entry() const;

        GameSettings &settings_;
        AgentValidator agent_validator_;
        Rml::DataModelHandle model_handle_;

        Rml::ElementDocument *document_ = nullptr;
        // removed from the document when Rml::Shutdown() destroys it, before
        // the members are torn down
        ExplorerNavListener explorer_nav_listener_;
        bool focus_explorer_pending_ = false;

        std::filesystem::path current_dir_;
        Rml::String algorithm_display_;
        Rml::String current_dir_display_;
        Rml::String agent_folder_display_;
        Rml::String agent_config_display_;
        // empty while no folder is chosen; otherwise success or error text
        Rml::String agent_status_;
        std::vector<Rml::String> entries_;
        bool agent_valid_ = false;
        bool can_play_ = false;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
