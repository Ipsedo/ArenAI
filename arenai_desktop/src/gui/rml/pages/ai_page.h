#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include "../../../core/agent_validator.h"
#include "../click_outside_listener.h"
#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    class AiPage final : public MenuPage {
    public:
        AiPage(
            MenuState &state, std::vector<AgentPreset> presets, AgentValidator validator,
            bool cuda_available);

        void after_context_update() override;

        bool on_back() override;

    protected:
        void bind_controls(DataModel &model) override;
        void on_attached(Rml::ElementDocument &document) override;

    private:
        enum class PendingFocus { None, FirstEntry, PresetHead, PresetList };

        class ExplorerNavListener final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event &event) override;
        };

        void refresh_explorer();
        bool close_dropdown();

        void select_preset(size_t index);
        void mark_custom();
        void validate_agent();
        void dirty_dropdown();

        void focus_first_entry() const;
        void focus_by_id(const char *id) const;
        void focus_in_list(const char *list_id) const;

        MenuState &state_;
        GameSettings &settings_;
        std::vector<AgentPreset> presets_;
        AgentValidator agent_validator_;
        bool cuda_available_;

        ExplorerNavListener explorer_nav_listener_;
        ClickOutsideListener click_outside_listener_;
        PendingFocus pending_focus_ = PendingFocus::None;

        std::optional<size_t> selected_preset_;
        std::vector<Rml::String> preset_names_;
        Rml::String preset_display_;
        bool preset_open_ = false;

        std::filesystem::path current_dir_;
        Rml::String algorithm_display_;
        Rml::String current_dir_display_;
        Rml::String agent_folder_display_;
        Rml::String agent_config_display_;
        Rml::String agent_status_;
        std::vector<Rml::String> entries_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_AI_PAGE_H
