//
// Created by samuel on 17/07/2026.
//

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_controller/callback.h>
#include <arenai_view/window.h>

#include "./menu.h"
#include "./rml/adapters.h"
#include "./rml/cursor_ring.h"
#include "./rml/hit_marker.h"
#include "./rml/input.h"
#include "./rml/reticle.h"

namespace arenai::desktop::gui {

    namespace {

        class RmlGui final : public AbstractGui {
        public:
            RmlGui(
                const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
                const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
                const GameSettings &initial_settings, const int window_width,
                const int window_height, SacFolderValidator sac_validator)
                : backend_(backend), window_(backend->get_window()), settings_(initial_settings),
                  width_(window_width), height_(window_height), file_interface_(asset_reader),
                  sac_validator_(std::move(sac_validator)),
                  controller_display_(
                      initial_settings.controller_kind == ControllerKind::Gamepad ? "gamepad"
                                                                                  : "keyboard"),
                  display_display_(initial_settings.fullscreen ? "fullscreen" : "windowed") {
                Rml::SetSystemInterface(&system_interface_);
                Rml::SetFileInterface(&file_interface_);
                Rml::SetRenderInterface(&backend_->ui_render_interface());
                Rml::Initialise();

                // menu.rcss draws the slider knob's detached cursor ring with
                // this gui-local decorator. Built only now: its property
                // registration needs the style-sheet specification that
                // Rml::Initialise() just created, so it cannot be a plain
                // member (members are constructed before this body runs).
                cursor_ring_instancer_ = std::make_unique<CursorRingDecoratorInstancer>();
                Rml::Factory::RegisterDecoratorInstancer(
                    "cursor-ring", cursor_ring_instancer_.get());
                hit_marker_instancer_ = std::make_unique<HitMarkerDecoratorInstancer>();
                Rml::Factory::RegisterDecoratorInstancer("hit-marker", hit_marker_instancer_.get());
                reticle_instancer_ = std::make_unique<ReticleDecoratorInstancer>();
                Rml::Factory::RegisterDecoratorInstancer("reticle", reticle_instancer_.get());

                load_fonts(asset_reader);

                context_ = Rml::CreateContext("menu", Rml::Vector2i(width_, height_));
                if (!context_) throw std::runtime_error("RmlUi context creation failed");
                update_dp_ratio();

                current_dir_ = std::filesystem::exists(settings_.sac_folder)
                                   ? std::filesystem::canonical(settings_.sac_folder)
                                   : std::filesystem::current_path();

                // the folder persisted from the previous run gets the same
                // dry-run check as a freshly picked one
                validate_sac_folder();

                bind_data_model();
                refresh_explorer();

                main_document_ = context_->LoadDocument("menu/main_menu.rml");
                params_document_ = context_->LoadDocument("menu/parameters.rml");
                pause_document_ = context_->LoadDocument("menu/pause.rml");
                game_over_document_ = context_->LoadDocument("menu/game_over.rml");
                hud_document_ = context_->LoadDocument("menu/hud.rml");
                if (!main_document_ || !params_document_ || !pause_document_ || !game_over_document_
                    || !hud_document_)
                    throw std::runtime_error("RmlUi menu documents failed to load");

                // D-pad bridge across the file explorer's scroll container
                reticle_ = hud_document_->GetElementById("reticle");
                hit_marker_ = hud_document_->GetElementById("hit-marker");
                if (reticle_ == nullptr || hit_marker_ == nullptr)
                    throw std::runtime_error("hud.rml misses its #reticle / #hit-marker elements");

                hud_document_->Show(Rml::ModalFlag::None, Rml::FocusFlag::None);

                params_document_->AddEventListener(Rml::EventId::Keydown, &explorer_nav_listener_);

                input_adapter_ = std::make_shared<MenuInputAdapter>(
                    context_,
                    [this] {
                        // Escape / gamepad B back out of the parameters screen;
                        // while paused B resumes the game (the application
                        // intercepts Escape itself before this adapter); the
                        // game-over popup cannot be backed out of
                        if (game_over_document_->IsVisible()) return;
                        if (pause_document_->IsVisible())
                            pending_pause_action_ = PauseAction::Continue;
                        else if (params_document_->IsVisible()) close_params();
                    },
                    [this](const bool gamepad) {
                        // menu.rcss shows the :focus highlight only under
                        // .gamepad-nav, so the mouse hover and the gamepad
                        // cursor are never visible together
                        for (auto *document:
                             {main_document_, params_document_, pause_document_,
                              game_over_document_})
                            document->SetClass("gamepad-nav", gamepad);
                    });
            }

            MenuOutcome run_main_menu() override {
                play_clicked_ = false;
                quit_clicked_ = false;

                window_->set_keyboard_callback(input_adapter_);
                window_->set_gamepad_callback(input_adapter_);
                window_->set_cursor_mode(controller::CursorMode::Normal);
                main_document_->Show();

                while (!window_->should_close() && !play_clicked_ && !quit_clicked_) {
                    window_->poll_events();

                    context_->Update();

                    // entering a directory rebuilt the entry clones during
                    // Update (dropping the focused one): put the cursor back
                    // on the first entry of the fresh listing so the gamepad
                    // walk resumes there — invisible for the mouse, the
                    // :focus highlight only shows under .gamepad-nav
                    if (std::exchange(focus_explorer_pending_, false)) focus_first_entry();

                    backend_->begin_ui_frame(width_, height_);
                    context_->Render();
                    backend_->end_ui_frame();
                    backend_->present();
                }

                main_document_->Hide();
                params_document_->Hide();
                window_->set_keyboard_callback(nullptr);
                window_->set_gamepad_callback(nullptr);

                return play_clicked_ ? MenuOutcome::Play : MenuOutcome::Quit;
            }

            GameSettings settings() const override { return settings_; }

            void open_pause(const int score) override {
                pending_pause_action_ = PauseAction::None;
                score_ = score;
                model_handle_.DirtyVariable("score");
                pause_document_->Show();
            }

            void close_pause() override { pause_document_->Hide(); }

            void open_game_over(const int score) override {
                pending_pause_action_ = PauseAction::None;
                score_ = score;
                model_handle_.DirtyVariable("score");
                game_over_document_->Show();
            }

            void close_game_over() override { game_over_document_->Hide(); }

            void render_pause_overlay() override {
                context_->Update();

                backend_->begin_ui_overlay(width_, height_);
                context_->Render();
                backend_->end_ui_frame();
            }

            PauseAction poll_pause_action() override {
                return std::exchange(pending_pause_action_, PauseAction::None);
            }

            void notify_hit(const HitKind kind) override {
                const bool kill = kind == HitKind::Kill;
                hit_marker_->SetClass("kill", kill);

                // Battlefield-style feedback: the ticks spread outward while
                // fading; a kill starts bigger, flares wider and lasts longer
                const float duration = kill ? KILL_MARKER_FADE_SECONDS : HIT_MARKER_FADE_SECONDS;
                const Rml::Tween tween(Rml::Tween::Quadratic, Rml::Tween::Out);

                const Rml::Property opaque(1.f, Rml::Unit::NUMBER);
                hit_marker_->Animate(
                    "opacity", Rml::Property(0.f, Rml::Unit::NUMBER), duration, tween, 1, false,
                    0.f, &opaque);

                const Rml::Property start_scale = Rml::Transform::MakeProperty(
                    {Rml::Transforms::Scale2D(kill ? KILL_MARKER_START_SCALE : 1.f)});
                hit_marker_->Animate(
                    "transform",
                    Rml::Transform::MakeProperty({Rml::Transforms::Scale2D(
                        kill ? KILL_MARKER_END_SCALE : HIT_MARKER_END_SCALE)}),
                    duration, tween, 1, false, 0.f, &start_scale);
            }

            void set_aim_point(const std::optional<glm::vec2> normalized) override {
                if (!normalized) {
                    reticle_->SetProperty(
                        Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Hidden));
                    return;
                }
                reticle_->SetProperty(
                    Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Visible));
                reticle_->SetProperty(
                    Rml::PropertyId::Left,
                    Rml::Property(normalized->x * static_cast<float>(width_), Rml::Unit::PX));
                reticle_->SetProperty(
                    Rml::PropertyId::Top,
                    Rml::Property(normalized->y * static_cast<float>(height_), Rml::Unit::PX));
            }

            void render_hud_overlay() override {
                context_->Update();

                backend_->begin_ui_overlay(width_, height_);
                context_->Render();
                backend_->end_ui_frame();
            }

            std::shared_ptr<controller::AbstractKeyboardCallback> pause_input() override {
                return input_adapter_;
            }

            std::shared_ptr<controller::AbstractGamepadCallback> pause_gamepad_input() override {
                return input_adapter_;
            }

            void on_window_resized(const int width, const int height) override {
                width_ = width;
                height_ = height;
                context_->SetDimensions(Rml::Vector2i(width_, height_));
                update_dp_ratio();
            }

            ~RmlGui() override {
                // nothing may keep pointing at this object through the window
                window_->set_keyboard_callback(nullptr);
                window_->set_gamepad_callback(nullptr);
                window_->set_resize_callback(nullptr);

                // releases the GL resources through the backend's render
                // interface, whose context is still current on this thread
                Rml::Shutdown();
            }

        private:
            // Every dp length in menu.rcss is mapped to pixels relative to a
            // 1080p design baseline, measured against the monitor the window
            // sits on — not the window itself — so the menu keeps the same
            // physical size on the display whether the game is fullscreen or
            // in a small window (a 4K TV renders it twice as large either
            // way). The min of both axes keeps the design fitting on unusual
            // ratios.
            void update_dp_ratio() const {
                const auto [screen_width, screen_height] = window_->screen_size();
                context_->SetDensityIndependentPixelRatio(std::max(
                    0.5f, std::min(
                              static_cast<float>(screen_width) / 1920.0f,
                              static_cast<float>(screen_height) / 1080.0f)));
            }

            // Registered with an explicit family/weight (the static TTFs carry
            // per-weight legacy family names that would not match the RCSS
            // font-family otherwise). The buffers must outlive Rml::Shutdown().
            void
            load_fonts(const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader) {
                struct FontSpec {
                    const char *path;
                    const char *family;
                    int weight;
                };
                constexpr FontSpec MENU_FONTS[] = {
                    {.path = "font/Sora-Regular.ttf", .family = "Sora", .weight = 400},
                    {.path = "font/Sora-SemiBold.ttf", .family = "Sora", .weight = 600},
                    {.path = "font/Sora-Bold.ttf", .family = "Sora", .weight = 700},
                    {.path = "font/IBMPlexMono-Regular.ttf",
                     .family = "IBM Plex Mono",
                     .weight = 400},
                    {.path = "font/IBMPlexMono-Medium.ttf",
                     .family = "IBM Plex Mono",
                     .weight = 500},
                    {.path = "font/IBMPlexMono-SemiBold.ttf",
                     .family = "IBM Plex Mono",
                     .weight = 600},
                };

                font_buffers_.reserve(std::size(MENU_FONTS));
                for (const auto &[path, family, weight]: MENU_FONTS) {
                    font_buffers_.push_back(asset_reader->read_text(path));
                    const auto &buffer = font_buffers_.back();
                    Rml::LoadFontFace(
                        Rml::Span(
                            reinterpret_cast<const Rml::byte *>(buffer.data()), buffer.size()),
                        family, Rml::Style::FontStyle::Normal,
                        static_cast<Rml::Style::FontWeight>(weight));
                }
            }

            void bind_data_model() {
                Rml::DataModelConstructor constructor = context_->CreateDataModel("settings");
                if (!constructor) throw std::runtime_error("RmlUi data model creation failed");

                constructor.RegisterArray<std::vector<Rml::String>>();

                constructor.Bind("nb_tanks", &settings_.nb_tanks);
                constructor.Bind("spawn_side", &settings_.spawn_side);
                constructor.Bind("controller", &controller_display_);
                constructor.Bind("display", &display_display_);
                constructor.Bind("sac_folder", &sac_folder_display_);
                constructor.Bind("sac_status", &sac_status_);
                constructor.Bind("sac_valid", &sac_valid_);
                constructor.Bind("current_dir", &current_dir_display_);
                constructor.Bind("entries", &entries_);
                constructor.Bind("can_play", &can_play_);
                constructor.Bind("score", &score_);

                constructor.BindEventCallback(
                    "play", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        if (can_play_) play_clicked_ = true;
                    });
                constructor.BindEventCallback(
                    "exit", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        quit_clicked_ = true;
                    });
                constructor.BindEventCallback(
                    "open_params",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        main_document_->Hide();
                        params_document_->Show();
                    });
                constructor.BindEventCallback(
                    "back", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        close_params();
                    });
                constructor.BindEventCallback(
                    "enter_dir",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (arguments.empty()) return;
                        const auto index = static_cast<size_t>(arguments[0].Get<int>());
                        if (index >= entries_.size()) return;

                        const std::string &entry = entries_[index];
                        current_dir_ =
                            entry == ".." ? current_dir_.parent_path() : current_dir_ / entry;
                        refresh_explorer();
                        focus_explorer_pending_ = true;
                    });
                constructor.BindEventCallback(
                    "set_controller",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (arguments.empty()) return;
                        controller_display_ = arguments[0].Get<Rml::String>();
                        settings_.controller_kind = controller_display_ == "gamepad"
                                                        ? ControllerKind::Gamepad
                                                        : ControllerKind::Keyboard;
                        model_handle_.DirtyVariable("controller");
                    });
                constructor.BindEventCallback(
                    "set_display",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (arguments.empty()) return;
                        display_display_ = arguments[0].Get<Rml::String>();
                        settings_.fullscreen = display_display_ == "fullscreen";
                        // applied immediately; the window reports its new size
                        // through the resize callback (dp-ratio included)
                        window_->set_fullscreen(settings_.fullscreen);
                        model_handle_.DirtyVariable("display");
                    });
                constructor.BindEventCallback(
                    "select_folder",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        settings_.sac_folder = current_dir_;
                        validate_sac_folder();
                        refresh_explorer();
                    });

                constructor.BindEventCallback(
                    "pause_continue",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        pending_pause_action_ = PauseAction::Continue;
                    });
                constructor.BindEventCallback(
                    "pause_main_menu",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        pending_pause_action_ = PauseAction::MainMenu;
                    });
                constructor.BindEventCallback(
                    "pause_exit",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        pending_pause_action_ = PauseAction::ExitGame;
                    });
                constructor.BindEventCallback(
                    "game_over_retry",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        pending_pause_action_ = PauseAction::Retry;
                    });

                model_handle_ = constructor.GetModelHandle();
            }

            void refresh_explorer() {
                entries_.clear();
                if (current_dir_.has_parent_path() && current_dir_ != current_dir_.root_path())
                    entries_.emplace_back("..");

                std::error_code list_error;
                for (const auto &entry:
                     std::filesystem::directory_iterator(current_dir_, list_error))
                    if (std::error_code type_error; entry.is_directory(type_error))
                        entries_.push_back(entry.path().filename().string());
                if (list_error)
                    std::cerr << "Cannot list " << current_dir_ << ": " << list_error.message()
                              << std::endl;

                // keep ".." pinned first, sort the actual directories
                const auto first_dir =
                    entries_.begin() + (!entries_.empty() && entries_[0] == ".." ? 1 : 0);
                std::sort(first_dir, entries_.end());

                current_dir_display_ = current_dir_.string();
                sac_folder_display_ = settings_.sac_folder.string();

                if (model_handle_) {
                    model_handle_.DirtyVariable("entries");
                    model_handle_.DirtyVariable("current_dir");
                    model_handle_.DirtyVariable("sac_folder");
                    model_handle_.DirtyVariable("sac_status");
                    model_handle_.DirtyVariable("sac_valid");
                    model_handle_.DirtyVariable("can_play");
                }
            }

            // runs the injected dry-run load and turns its outcome into the
            // tri-state the parameters screen displays (nothing chosen yet /
            // model loaded / error message); can_play_ follows the real load
            void validate_sac_folder() {
                if (settings_.sac_folder.empty()) {
                    sac_valid_ = false;
                    sac_status_ = "";
                } else {
                    const auto error = sac_validator_(settings_.sac_folder);
                    sac_valid_ = !error.has_value();
                    sac_status_ = error.value_or("AI model loaded");
                }
                can_play_ = sac_valid_;
            }

            void close_params() const {
                params_document_->Hide();
                main_document_->Show();
            }

            void focus_first_entry() const {
                const Rml::Element *list = params_document_->GetElementById("file-list");
                if (list == nullptr) return;
                const auto entries = ExplorerNavListener::visible_file_entries(list);
                if (entries.empty()) return;
                if (entries.front()->Focus(true))
                    entries.front()->ScrollIntoView(Rml::ScrollAlignment::Nearest);
            }

            std::shared_ptr<view::AbstractWindowedGraphicBackend> backend_;
            std::shared_ptr<view::AbstractWindow> window_;

            GameSettings settings_;
            int width_;
            int height_;

            MenuSystemInterface system_interface_;
            ReaderBackedFileInterface file_interface_;
            // unique_ptr: created after Rml::Initialise(), and member
            // destruction keeps it alive until after Rml::Shutdown() as
            // RmlUi requires of registered instancers

            std::unique_ptr<CursorRingDecoratorInstancer> cursor_ring_instancer_;
            std::unique_ptr<HitMarkerDecoratorInstancer> hit_marker_instancer_;
            std::unique_ptr<ReticleDecoratorInstancer> reticle_instancer_;
            std::vector<std::string> font_buffers_;

            Rml::Context *context_ = nullptr;
            Rml::ElementDocument *main_document_ = nullptr;
            Rml::ElementDocument *params_document_ = nullptr;
            Rml::ElementDocument *pause_document_ = nullptr;
            Rml::ElementDocument *game_over_document_ = nullptr;
            Rml::ElementDocument *hud_document_ = nullptr;
            Rml::Element *reticle_ = nullptr;
            Rml::Element *hit_marker_ = nullptr;
            static constexpr float HIT_MARKER_FADE_SECONDS = 0.45f;
            static constexpr float HIT_MARKER_END_SCALE = 1.3f;
            static constexpr float KILL_MARKER_FADE_SECONDS = 0.6f;
            static constexpr float KILL_MARKER_START_SCALE = 1.1f;
            static constexpr float KILL_MARKER_END_SCALE = 1.55f;
            Rml::DataModelHandle model_handle_;

            std::shared_ptr<MenuInputAdapter> input_adapter_;
            // removed from the document when Rml::Shutdown() destroys it in
            // the destructor body, before the members are torn down
            ExplorerNavListener explorer_nav_listener_;
            bool focus_explorer_pending_ = false;

            std::filesystem::path current_dir_;
            SacFolderValidator sac_validator_;
            Rml::String controller_display_;
            Rml::String display_display_;
            Rml::String current_dir_display_;
            Rml::String sac_folder_display_;
            // empty while no folder is chosen; otherwise success or error text
            Rml::String sac_status_;
            std::vector<Rml::String> entries_;
            bool sac_valid_ = false;
            bool can_play_ = false;
            // score shown by the pause and game-over popups
            int score_ = 0;
            bool play_clicked_ = false;
            bool quit_clicked_ = false;
            PauseAction pending_pause_action_ = PauseAction::None;
        };

    }// namespace

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
        const GameSettings &initial_settings, const int window_width, const int window_height,
        SacFolderValidator sac_validator) {
        return std::make_unique<RmlGui>(
            backend, asset_reader, initial_settings, window_width, window_height,
            std::move(sac_validator));
    }

}// namespace arenai::desktop::gui
