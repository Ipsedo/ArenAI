//
// Created by samuel on 06/09/2026.
//

#include "./ai_page.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace arenai::desktop::gui {

    namespace {
        constexpr const char *CUSTOM_PRESET = "Custom";

        // element ids of ai.rml the page drives from code
        constexpr const char *FILE_LIST_ID = "file-list";
        constexpr const char *PRESET_HEAD_ID = "preset-head";
        constexpr const char *PRESET_LIST_ID = "preset-list";
        constexpr const char *CPU_TOGGLE_ID = "cpu-toggle";
        constexpr const char *CUDA_TOGGLE_ID = "cuda-toggle";
        constexpr const char *USE_FOLDER_ID = "use-folder";

        // the data-for clones currently shown in the explorer
        std::vector<Rml::Element *> visible_file_entries(const Rml::Element *file_list) {
            std::vector<Rml::Element *> entries;
            for (int i = 0; i < file_list->GetNumChildren(); i++)
                if (Rml::Element *child = file_list->GetChild(i); child->IsVisible())
                    entries.push_back(child);
            return entries;
        }
    }// namespace

    void AiPage::ExplorerNavListener::ProcessEvent(Rml::Event &event) {
        const auto key = static_cast<Rml::Input::KeyIdentifier>(
            event.GetParameter<int>("key_identifier", Rml::Input::KI_UNKNOWN));
        if (key != Rml::Input::KI_UP && key != Rml::Input::KI_DOWN) return;

        const Rml::Element *focus = event.GetTargetElement();
        Rml::ElementDocument *document = focus->GetOwnerDocument();
        if (document == nullptr) return;
        const Rml::Element *list = document->GetElementById(FILE_LIST_ID);
        Rml::Element *above = document->GetElementById(CUDA_TOGGLE_ID);
        const Rml::Element *above_cpu = document->GetElementById(CPU_TOGGLE_ID);
        Rml::Element *below = document->GetElementById(USE_FOLDER_ID);
        if (list == nullptr || above == nullptr || above_cpu == nullptr || below == nullptr) return;

        const auto entries = visible_file_entries(list);
        if (entries.empty()) return;

        Rml::Element *target = nullptr;
        if (focus->GetParentNode() == list) {
            if (key == Rml::Input::KI_UP && focus == entries.front()) target = above;
            else if (key == Rml::Input::KI_DOWN && focus == entries.back()) target = below;
        } else if (key == Rml::Input::KI_DOWN && (focus == above || focus == above_cpu))
            target = entries.front();
        else if (key == Rml::Input::KI_UP && focus == below) target = entries.back();

        if (target != nullptr && target->Focus(true)) {
            target->ScrollIntoView(Rml::ScrollAlignment::Nearest);
            event.StopPropagation();
        }
    }

    AiPage::AiPage(
        MenuState &state, std::vector<AgentPreset> presets, AgentValidator validator,
        const bool cuda_available)
        : MenuPage(PageId::Ai, PageId::Parameters, "menu/ai.rml", "ai"), state_(state),
          settings_(state.settings), presets_(std::move(presets)),
          agent_validator_(std::move(validator)), cuda_available_(cuda_available),
          click_outside_listener_([this] { close_dropdown(); }),
          algorithm_display_(to_string(settings_.agent_algorithm)) {
        for (const auto &preset: presets_) preset_names_.push_back(preset.name);

        // a preference saved on a machine that had CUDA never reaches the
        // loader on one that does not
        if (!cuda_available_) settings_.cuda = false;

        // the preset persisted from the previous run, matched by name; an
        // unknown name (or none) means a hand-picked selection
        if (settings_.agent_preset)
            if (const auto found =
                    std::find(preset_names_.begin(), preset_names_.end(), *settings_.agent_preset);
                found != preset_names_.end())
                selected_preset_ = static_cast<size_t>(found - preset_names_.begin());
        if (!selected_preset_) settings_.agent_preset = std::nullopt;
        preset_display_ = selected_preset_ ? preset_names_[*selected_preset_] : CUSTOM_PRESET;

        current_dir_ = std::filesystem::exists(settings_.agent_folder)
                           ? std::filesystem::canonical(settings_.agent_folder)
                           : std::filesystem::current_path();

        // the selection persisted from the previous run gets the same
        // dry-run check as a freshly picked one
        validate_agent();
    }

    void AiPage::bind_controls(DataModel &model) {
        model.register_array<std::vector<Rml::String>>();

        model.bind("presets", &preset_names_);
        model.bind("preset", &preset_display_);
        model.bind("preset_open", &preset_open_);
        model.bind("algorithm", &algorithm_display_);
        model.bind("agent_folder", &agent_folder_display_);
        model.bind("agent_config", &agent_config_display_);
        model.bind("agent_status", &agent_status_);
        model.bind("agent_valid", &state_.agent_valid);
        model.bind("cuda", &settings_.cuda);
        model.bind("cuda_available", &cuda_available_);
        model.bind("current_dir", &current_dir_display_);
        model.bind("entries", &entries_);

        model.on_event("toggle_preset", [this](const Rml::VariantList &) {
            preset_open_ = !preset_open_;
            pending_focus_ = preset_open_ ? PendingFocus::PresetList : PendingFocus::PresetHead;
            dirty_dropdown();
        });
        model.on_event("set_preset", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const auto index = static_cast<size_t>(arguments[0].Get<int>());
            if (index >= presets_.size()) return;
            select_preset(index);
            preset_open_ = false;
            pending_focus_ = PendingFocus::PresetHead;
            refresh_explorer();
        });
        model.on_event("set_algorithm", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const auto algorithm = ai_algorithm_from_string(arguments[0].Get<Rml::String>());
            if (!algorithm) return;
            if (settings_.agent_algorithm != *algorithm) {
                settings_.agent_algorithm = *algorithm;
                algorithm_display_ = to_string(*algorithm);
                mark_custom();
                // the networks are rebuilt differently: re-run the check
                validate_agent();
            }
            refresh_explorer();
        });
        model.on_event("select_entry", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const auto index = static_cast<size_t>(arguments[0].Get<int>());
            if (index >= entries_.size()) return;

            const std::string &entry = entries_[index];
            if (entry == "..") current_dir_ = current_dir_.parent_path();
            else if (const auto path = current_dir_ / entry; std::filesystem::is_directory(path))
                current_dir_ = path;
            else {
                // a listed file is a config.json candidate
                settings_.agent_config = path;
                mark_custom();
                validate_agent();
                refresh_explorer();
                return;
            }
            refresh_explorer();
            pending_focus_ = PendingFocus::FirstEntry;
        });
        model.on_event("set_device", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const bool cuda = arguments[0].Get<Rml::String>() == "cuda";
            if (cuda && !cuda_available_) return;
            if (settings_.cuda == cuda) return;
            settings_.cuda = cuda;
            this->model().dirty("cuda");
        });
        model.on_event("select_folder", [this](const Rml::VariantList &) {
            settings_.agent_folder = current_dir_;
            mark_custom();
            validate_agent();
            refresh_explorer();
        });
    }

    void AiPage::on_attached(Rml::ElementDocument &document) {
        document.AddEventListener(Rml::EventId::Keydown, &explorer_nav_listener_);
        document.AddEventListener(Rml::EventId::Click, &click_outside_listener_);
        refresh_explorer();
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

        model().dirty(
            {"entries", "current_dir", "agent_folder", "agent_config", "agent_status",
             "agent_valid", "algorithm"});
        dirty_dropdown();
    }

    void AiPage::after_context_update() {
        // the data-if / data-for clones were rebuilt during Update, dropping
        // the focused one: put the cursor back where the action leads —
        // invisible for the mouse, the :focus highlight only shows under
        // .gamepad-nav
        switch (std::exchange(pending_focus_, PendingFocus::None)) {
            case PendingFocus::FirstEntry: focus_first_entry(); break;
            case PendingFocus::PresetHead: focus_by_id(PRESET_HEAD_ID); break;
            case PendingFocus::PresetList: focus_in_list(PRESET_LIST_ID); break;
            case PendingFocus::None: break;
        }
    }

    bool AiPage::on_back() { return close_dropdown(); }

    bool AiPage::close_dropdown() {
        if (!preset_open_) return false;
        pending_focus_ = PendingFocus::PresetHead;
        preset_open_ = false;
        dirty_dropdown();
        return true;
    }

    // copies the preset's tuple into the settings and moves the explorer
    // onto its state-dict folder
    void AiPage::select_preset(const size_t index) {
        apply_preset(presets_[index], settings_);
        selected_preset_ = index;
        preset_display_ = preset_names_[index];
        algorithm_display_ = to_string(settings_.agent_algorithm);
        if (std::filesystem::is_directory(settings_.agent_folder))
            current_dir_ = std::filesystem::canonical(settings_.agent_folder);
        validate_agent();
    }

    // any hand edit of the agent fields detaches the selection from the preset
    void AiPage::mark_custom() {
        selected_preset_ = std::nullopt;
        settings_.agent_preset = std::nullopt;
        preset_display_ = CUSTOM_PRESET;
    }

    void AiPage::dirty_dropdown() { model().dirty({"preset", "preset_open"}); }

    // runs the injected dry-run load and turns its outcome into the
    // tri-state the AI screen displays (nothing chosen yet / model
    // loaded / error message); the Play gate follows the real load
    void AiPage::validate_agent() {
        if (settings_.agent_folder.empty()) {
            state_.agent_valid = false;
            agent_status_ = "";
            return;
        }
        const auto error = agent_validator_(agent_selection(settings_));
        state_.agent_valid = !error.has_value();
        agent_status_ = error.value_or("AI model loaded");
    }

    void AiPage::focus_first_entry() const {
        const Rml::Element *list = document().GetElementById(FILE_LIST_ID);
        if (list == nullptr) return;
        const auto entries = visible_file_entries(list);
        if (entries.empty()) return;
        if (entries.front()->Focus(true))
            entries.front()->ScrollIntoView(Rml::ScrollAlignment::Nearest);
    }

    void AiPage::focus_by_id(const char *id) const {
        if (Rml::Element *element = document().GetElementById(id)) element->Focus(true);
    }

    void AiPage::focus_in_list(const char *list_id) const {
        const Rml::Element *list = document().GetElementById(list_id);
        if (list == nullptr || list->GetNumChildren() == 0) return;
        Rml::Element *target = list->GetChild(0);
        for (int i = 0; i < list->GetNumChildren(); ++i)
            if (Rml::Element *child = list->GetChild(i); child->IsClassSet("selected")) {
                target = child;
                break;
            }
        target->Focus(true);
    }

}// namespace arenai::desktop::gui
