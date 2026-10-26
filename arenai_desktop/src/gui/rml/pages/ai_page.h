//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include "../../../core/agent_validator.h"
#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // The AI screen: the preset dropdown, the algorithm dropdown and the
    // file explorer picking the trained model's folder, dry-run validated
    // through the injected AgentValidator. Writes the agent fields of the
    // shared settings and the agent_valid flag the other screens display.
    class AiPage final : public MenuPage {
    public:
        AiPage(
            MenuState &state, std::vector<AgentPreset> presets, AgentValidator validator,
            bool cuda_available);

        // after the context Update rebuilt the data-bound clones, put the
        // gamepad cursor back where the last action left it (the fresh
        // explorer listing, the just-opened list, the closed dropdown head)
        void after_context_update() override;

        // Escape / gamepad B folds an open dropdown first
        bool on_back() override;

    protected:
        void bind_controls(DataModel &model) override;
        // hooks the D-pad bridge across the explorer's scroll container and
        // the click-outside closing of the dropdowns
        void on_attached(Rml::ElementDocument &document) override;

    private:
        // which element the gamepad cursor lands on after the next Update
        enum class PendingFocus {
            None,
            FirstEntry,
            PresetHead,
            PresetList,
            AlgorithmHead,
            AlgorithmList
        };

        class ClickOutsideListener final : public Rml::EventListener {
        public:
            explicit ClickOutsideListener(std::function<void()> on_outside);
            void ProcessEvent(Rml::Event &event) override;

        private:
            std::function<void()> on_outside_;
        };

        // D-pad bridge across the file explorer: RmlUi's spatial navigation
        // stops at the scroll container, so Up from the first entry lands on
        // the algorithm dropdown head, Down from the last one on the "Use
        // this folder" button, and back
        class ExplorerNavListener final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event &event) override;
        };

        void refresh_explorer();
        bool close_dropdowns();

        void select_preset(size_t index);
        void mark_custom();
        void validate_agent();
        void dirty_dropdowns();

        void focus_first_entry() const;
        void focus_by_id(const char *id) const;
        // the selected item of an open list, else its first one
        void focus_in_list(const char *list_id) const;

        MenuState &state_;
        GameSettings &settings_;
        std::vector<AgentPreset> presets_;
        AgentValidator agent_validator_;
        // no CUDA device on this machine: the toggle shows disabled and off
        bool cuda_available_;

        // removed from the document when Rml::Shutdown() destroys it, before
        // the members are torn down
        ExplorerNavListener explorer_nav_listener_;
        ClickOutsideListener click_outside_listener_;
        PendingFocus pending_focus_ = PendingFocus::None;

        // nullopt = "Custom": the agent fields no longer match a preset
        std::optional<size_t> selected_preset_;
        std::vector<Rml::String> preset_names_;
        Rml::String preset_display_;
        bool preset_open_ = false;
        bool algorithm_open_ = false;

        std::filesystem::path current_dir_;
        Rml::String algorithm_display_;
        Rml::String current_dir_display_;
        Rml::String agent_folder_display_;
        Rml::String agent_config_display_;
        // empty while no folder is chosen; otherwise success or error text
        Rml::String agent_status_;
        std::vector<Rml::String> entries_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
