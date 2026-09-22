//
// Created by samuel on 06/09/2026.
//

#include "./ai_page.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace arenai::desktop::gui {

    AiPage::AiPage(GameSettings &settings, AgentValidator validator)
        : settings_(settings), agent_validator_(std::move(validator)),
          algorithm_display_(to_string(settings.agent_algorithm)) {
        current_dir_ = std::filesystem::exists(settings_.agent_folder)
                           ? std::filesystem::canonical(settings_.agent_folder)
                           : std::filesystem::current_path();

        // the selection persisted from the previous run gets the same
        // dry-run check as a freshly picked one
        validate_agent();
    }

    void AiPage::bind(Rml::DataModelConstructor &constructor) {
        constructor.Bind("algorithm", &algorithm_display_);
        constructor.Bind("agent_folder", &agent_folder_display_);
        constructor.Bind("agent_config", &agent_config_display_);
        constructor.Bind("agent_status", &agent_status_);
        constructor.Bind("agent_valid", &agent_valid_);
        constructor.Bind("current_dir", &current_dir_display_);
        constructor.Bind("entries", &entries_);
        constructor.Bind("can_play", &can_play_);

        constructor.BindEventCallback(
            "select_entry",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (arguments.empty()) return;
                const auto index = static_cast<size_t>(arguments[0].Get<int>());
                if (index >= entries_.size()) return;

                const std::string &entry = entries_[index];
                if (entry == "..") current_dir_ = current_dir_.parent_path();
                else if (const auto path = current_dir_ / entry;
                         std::filesystem::is_directory(path))
                    current_dir_ = path;
                else {
                    // a listed file is a config.json candidate
                    settings_.agent_config = path;
                    validate_agent();
                    refresh_explorer();
                    return;
                }
                refresh_explorer();
                focus_explorer_pending_ = true;
            });
        constructor.BindEventCallback(
            "set_algorithm",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (arguments.empty()) return;
                const auto algorithm = ai_algorithm_from_string(arguments[0].Get<Rml::String>());
                if (!algorithm) return;
                settings_.agent_algorithm = *algorithm;
                algorithm_display_ = to_string(*algorithm);
                model_handle_.DirtyVariable("algorithm");
                // the networks are rebuilt differently: re-run the check
                validate_agent();
                refresh_explorer();
            });
        constructor.BindEventCallback(
            "select_folder", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                settings_.agent_folder = current_dir_;
                validate_agent();
                refresh_explorer();
            });
    }

    void AiPage::set_model_handle(const Rml::DataModelHandle handle) { model_handle_ = handle; }

    void AiPage::attach(Rml::ElementDocument &document) {
        document_ = &document;
        document.AddEventListener(Rml::EventId::Keydown, &explorer_nav_listener_);
    }

    void AiPage::refresh_explorer() {
        entries_.clear();
        if (current_dir_.has_parent_path() && current_dir_ != current_dir_.root_path())
            entries_.emplace_back("..");

        std::vector<Rml::String> config_files;
        std::error_code list_error;
        for (const auto &entry: std::filesystem::directory_iterator(current_dir_, list_error))
            if (std::error_code type_error; entry.is_directory(type_error))
                entries_.push_back(entry.path().filename().string());
            else if (entry.path().extension() == ".json")
                config_files.push_back(entry.path().filename().string());
        if (list_error)
            std::cerr << "Cannot list " << current_dir_ << ": " << list_error.message()
                      << std::endl;

        // keep ".." pinned first, then the directories, then the
        // config.json candidates, each block sorted
        const auto first_dir =
            entries_.begin() + (!entries_.empty() && entries_[0] == ".." ? 1 : 0);
        std::sort(first_dir, entries_.end());
        std::sort(config_files.begin(), config_files.end());
        entries_.insert(entries_.end(), config_files.begin(), config_files.end());

        current_dir_display_ = current_dir_.string();
        agent_folder_display_ = settings_.agent_folder.string();
        agent_config_display_ = settings_.agent_config.empty()
                                    ? "auto (config.json next to the state dicts)"
                                    : settings_.agent_config.string();

        if (model_handle_) {
            model_handle_.DirtyVariable("entries");
            model_handle_.DirtyVariable("current_dir");
            model_handle_.DirtyVariable("agent_folder");
            model_handle_.DirtyVariable("agent_config");
            model_handle_.DirtyVariable("agent_status");
            model_handle_.DirtyVariable("agent_valid");
            model_handle_.DirtyVariable("can_play");
        }
    }

    void AiPage::update_after_context() {
        // entering a directory rebuilt the entry clones during Update
        // (dropping the focused one): put the cursor back on the first entry
        // of the fresh listing so the gamepad walk resumes there — invisible
        // for the mouse, the :focus highlight only shows under .gamepad-nav
        if (std::exchange(focus_explorer_pending_, false)) focus_first_entry();
    }

    // runs the injected dry-run load and turns its outcome into the
    // tri-state the AI screen displays (nothing chosen yet / model
    // loaded / error message); can_play_ follows the real load
    void AiPage::validate_agent() {
        if (settings_.agent_folder.empty()) {
            agent_valid_ = false;
            agent_status_ = "";
        } else {
            const auto error = agent_validator_(
                {.config = settings_.agent_config,
                 .folder = settings_.agent_folder,
                 .algorithm = settings_.agent_algorithm});
            agent_valid_ = !error.has_value();
            agent_status_ = error.value_or("AI model loaded");
        }
        can_play_ = agent_valid_;
    }

    void AiPage::focus_first_entry() const {
        const Rml::Element *list = document_->GetElementById("file-list");
        if (list == nullptr) return;
        const auto entries = ExplorerNavListener::visible_file_entries(list);
        if (entries.empty()) return;
        if (entries.front()->Focus(true))
            entries.front()->ScrollIntoView(Rml::ScrollAlignment::Nearest);
    }

}// namespace arenai::desktop::gui
