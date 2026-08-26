//
// Created by samuel on 17/07/2026.
//

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_controller/callback.h>
#include <arenai_controller/input_names.h>
#include <arenai_view/window.h>

#include "./menu.h"
#include "./rml/adapters.h"
#include "./rml/cursor_ring.h"
#include "./rml/hit_marker.h"
#include "./rml/input.h"
#include "./rml/reticle.h"

namespace arenai::desktop::gui {

    namespace {

        // one row of the controls page: an action and its current binding
        struct BindingRow {
            Rml::String label;
            Rml::String binding;
            bool listening = false;
            bool bound = true;
        };

        // slot order of the two pages; the gamepad page puts its button slot
        // (fire) first, then the five axis slots
        constexpr const char *KB_SLOT_LABELS[] = {
            "FORWARD", "BACKWARD", "TURN LEFT", "TURN RIGHT", "FIRE"};
        constexpr const char *GP_SLOT_LABELS[] = {"FIRE",  "STEER",      "AIM X",
                                                  "AIM Y", "ACCELERATE", "REVERSE"};
        constexpr int NB_KB_SLOTS = 5;
        constexpr int NB_GP_SLOTS = 6;
        // accelerate / reverse read a single direction of their axis
        constexpr bool gp_slot_is_one_way(const int slot) { return slot >= 4; }

        // an axis must come back to rest before it can be captured: without
        // this the stick still deflected from navigating the menu (or the A
        // press bound to a trigger) would bind itself instantly
        constexpr double CAPTURE_ENGAGE = 0.6, CAPTURE_REST = 0.3;

        // a capture nobody feeds cancels itself: Escape is the only cancel
        // input (every pad button stays bindable), so a pad-only player
        // needs the timeout
        constexpr auto CAPTURE_TIMEOUT = std::chrono::seconds(15);

        constexpr const char *GAMEPAD_AXIS_LABELS[] = {"L-STICK X", "L-STICK Y", "R-STICK X",
                                                       "R-STICK Y", "LT",        "RT"};

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
                controls_document_ = context_->LoadDocument("menu/controls.rml");
                pause_document_ = context_->LoadDocument("menu/pause.rml");
                game_over_document_ = context_->LoadDocument("menu/game_over.rml");
                hud_document_ = context_->LoadDocument("menu/hud.rml");
                if (!main_document_ || !params_document_ || !controls_document_ || !pause_document_
                    || !game_over_document_ || !hud_document_)
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
                        // Escape / gamepad B back out of the controls and
                        // parameters screens; while paused B resumes the game
                        // (the application intercepts Escape itself before
                        // this adapter); the game-over popup cannot be backed
                        // out of
                        if (game_over_document_->IsVisible()) return;
                        if (pause_document_->IsVisible())
                            pending_pause_action_ = PauseAction::Continue;
                        else if (controls_document_->IsVisible()) close_controls();
                        else if (params_document_->IsVisible()) close_params();
                    },
                    [this](const bool gamepad) {
                        // menu.rcss shows the :focus highlight only under
                        // .gamepad-nav, so the mouse hover and the gamepad
                        // cursor are never visible together
                        for (auto *document:
                             {main_document_, params_document_, controls_document_, pause_document_,
                              game_over_document_})
                            document->SetClass("gamepad-nav", gamepad);
                    });

                // route the pad input to the pad persisted from the previous
                // session, when it is connected
                refresh_gamepad_list();
                rebuild_binding_rows();
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

                    // pads can be (un)plugged while the device list is on
                    // screen; the refresh is a no-op while nothing changed
                    if (controls_document_->IsVisible()
                        && settings_.controller_kind == ControllerKind::Gamepad)
                        refresh_gamepad_list();

                    // a forgotten capture ends on its own (a pad-only player
                    // has no Escape at hand)
                    if (capture_slot_ >= 0
                        && std::chrono::steady_clock::now() >= capture_deadline_) {
                        end_capture();
                        rebuild_binding_rows();
                    }

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
                end_capture();
                controls_document_->Hide();
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

                if (auto row_handle = constructor.RegisterStruct<BindingRow>()) {
                    row_handle.RegisterMember("label", &BindingRow::label);
                    row_handle.RegisterMember("binding", &BindingRow::binding);
                    row_handle.RegisterMember("listening", &BindingRow::listening);
                    row_handle.RegisterMember("bound", &BindingRow::bound);
                }
                constructor.RegisterArray<std::vector<BindingRow>>();

                constructor.Bind("kb_rows", &kb_rows_);
                constructor.Bind("gp_rows", &gp_rows_);
                constructor.Bind("gamepads", &gamepad_names_);
                constructor.Bind("selected_gamepad", &selected_gamepad_);
                constructor.Bind("bind_status", &bind_status_);
                constructor.Bind("bind_warning", &bind_warning_);

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
                    "open_controls",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        open_controls();
                    });
                constructor.BindEventCallback(
                    "controls_back",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        close_controls();
                    });
                constructor.BindEventCallback(
                    "capture_kb",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (!arguments.empty()) begin_capture(true, arguments[0].Get<int>());
                    });
                constructor.BindEventCallback(
                    "capture_gp",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (!arguments.empty()) begin_capture(false, arguments[0].Get<int>());
                    });
                constructor.BindEventCallback(
                    "select_pad",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                        if (arguments.empty()) return;
                        const auto index = static_cast<size_t>(arguments[0].Get<int>());
                        if (index >= gamepads_.size()) return;

                        settings_.bindings.gamepad.device_guid = gamepads_[index].guid;
                        settings_.bindings.gamepad.device_name = gamepads_[index].name;
                        window_->select_gamepad(gamepads_[index].id);
                        selected_gamepad_ = static_cast<int>(index);
                        model_handle_.DirtyVariable("selected_gamepad");
                    });
                constructor.BindEventCallback(
                    "reset_bindings",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        end_capture();
                        if (settings_.controller_kind == ControllerKind::Keyboard)
                            settings_.bindings.keyboard = {};
                        else {
                            // the device choice is not a binding: keep it
                            auto gamepad = GamepadBindings{};
                            gamepad.device_guid = std::move(settings_.bindings.gamepad.device_guid);
                            gamepad.device_name = std::move(settings_.bindings.gamepad.device_name);
                            settings_.bindings.gamepad = std::move(gamepad);
                        }
                        set_default_bind_status();
                        rebuild_binding_rows();
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

            // ---- controls page ------------------------------------------

            void open_controls() {
                params_document_->Hide();
                refresh_gamepad_list();
                set_default_bind_status();
                rebuild_binding_rows();
                controls_document_->Show();
            }

            void close_controls() {
                end_capture();
                rebuild_binding_rows();
                controls_document_->Hide();
                params_document_->Show();
            }

            std::array<std::optional<KeyboardBinding> *, NB_KB_SLOTS> kb_slots() {
                auto &keyboard = settings_.bindings.keyboard;
                return {
                    &keyboard.forward, &keyboard.backward, &keyboard.turn_left,
                    &keyboard.turn_right, &keyboard.fire};
            }

            // gamepad slots 1..5 (0 is the fire button)
            std::array<std::optional<GamepadAxisBinding> *, NB_GP_SLOTS - 1> gp_axis_slots() {
                auto &gamepad = settings_.bindings.gamepad;
                return {
                    &gamepad.steer, &gamepad.aim_x, &gamepad.aim_y, &gamepad.accelerate,
                    &gamepad.reverse};
            }

            Rml::String keyboard_slot_label(const std::optional<KeyboardBinding> &slot) const {
                if (!slot) return "UNBOUND";
                if (const auto *button = std::get_if<controller::MouseButton>(&*slot))
                    switch (*button) {
                        case controller::MouseButton::Left: return "MOUSE L";
                        case controller::MouseButton::Right: return "MOUSE R";
                        case controller::MouseButton::Middle: return "MOUSE M";
                    }
                // layout-aware label (Key::Q reads "A" on AZERTY)
                const auto label = window_->key_label(std::get<controller::Key>(*slot));
                return label.empty() ? "?" : Rml::String(label);
            }

            static Rml::String
            axis_slot_label(const std::optional<GamepadAxisBinding> &slot, const bool one_way) {
                if (!slot) return "UNBOUND";
                Rml::String label = GAMEPAD_AXIS_LABELS[static_cast<int>(slot->axis)];
                // a one-way action bound to a stick reads a single direction
                const bool on_stick = slot->axis != GamepadAxis::LeftTrigger
                                      && slot->axis != GamepadAxis::RightTrigger;
                if (one_way && on_stick) label += slot->sign > 0.f ? "+" : "-";
                return label;
            }

            void rebuild_binding_rows() {
                kb_rows_.resize(NB_KB_SLOTS);
                const auto keyboard_slots = kb_slots();
                for (int i = 0; i < NB_KB_SLOTS; i++) {
                    const bool listening = capture_keyboard_page_ && capture_slot_ == i;
                    kb_rows_[i] = {
                        .label = KB_SLOT_LABELS[i],
                        .binding =
                            listening ? "PRESS A KEY..." : keyboard_slot_label(*keyboard_slots[i]),
                        .listening = listening,
                        .bound = keyboard_slots[i]->has_value()};
                }

                gp_rows_.resize(NB_GP_SLOTS);
                const auto axis_slots = gp_axis_slots();
                for (int i = 0; i < NB_GP_SLOTS; i++) {
                    const bool listening = !capture_keyboard_page_ && capture_slot_ == i;
                    const bool fire_slot = i == 0;
                    Rml::String binding;
                    if (listening) binding = fire_slot ? "PRESS A BUTTON..." : "MOVE AN AXIS...";
                    else if (fire_slot)
                        binding = settings_.bindings.gamepad.fire
                                      ? controller::to_string(*settings_.bindings.gamepad.fire)
                                      : "UNBOUND";
                    else binding = axis_slot_label(*axis_slots[i - 1], gp_slot_is_one_way(i));
                    gp_rows_[i] = {
                        .label = GP_SLOT_LABELS[i],
                        .binding = std::move(binding),
                        .listening = listening,
                        .bound = fire_slot ? settings_.bindings.gamepad.fire.has_value()
                                           : axis_slots[i - 1]->has_value()};
                }

                if (model_handle_) {
                    model_handle_.DirtyVariable("kb_rows");
                    model_handle_.DirtyVariable("gp_rows");
                    model_handle_.DirtyVariable("bind_status");
                    model_handle_.DirtyVariable("bind_warning");
                }
            }

            void set_default_bind_status() {
                bind_warning_ = false;
                bind_status_ = settings_.controller_kind == ControllerKind::Keyboard
                                   ? "Click a slot, then press the new key or mouse button. "
                                     "Escape cancels."
                                   : "Click a slot, then press a button or move an axis. "
                                     "Escape or 15s of inactivity cancels — Start stays the "
                                     "pause toggle.";
            }

            void begin_capture(const bool keyboard_page, const int slot) {
                const int nb_slots = keyboard_page ? NB_KB_SLOTS : NB_GP_SLOTS;
                if (slot < 0 || slot >= nb_slots) return;

                capture_keyboard_page_ = keyboard_page;
                capture_slot_ = slot;
                capture_deadline_ = std::chrono::steady_clock::now() + CAPTURE_TIMEOUT;
                // every axis must return to rest once before it can bind (the
                // stick may still be deflected from navigating the menu)
                axis_armed_.fill(false);
                set_default_bind_status();

                input_adapter_->set_capture_sink(
                    [this](const RawMenuInput &input) { on_capture_input(input); });
                rebuild_binding_rows();
            }

            void end_capture() {
                capture_slot_ = -1;
                input_adapter_->set_capture_sink(nullptr);
            }

            void on_capture_input(const RawMenuInput &input) {
                // Escape is the only cancel input (every pad button stays
                // bindable); a pad-only player relies on the capture timeout
                if (const auto *key = std::get_if<controller::Key>(&input);
                    key != nullptr && *key == controller::Key::Escape) {
                    end_capture();
                    rebuild_binding_rows();
                    return;
                }

                if (capture_keyboard_page_) {
                    if (const auto *key = std::get_if<controller::Key>(&input))
                        assign_keyboard(KeyboardBinding(*key));
                    else if (const auto *button = std::get_if<controller::MouseButton>(&input))
                        assign_keyboard(KeyboardBinding(*button));
                    // pad input has no meaning on the keyboard page
                    return;
                }

                if (capture_slot_ == 0) {
                    // Start stays the in-game pause toggle, never a binding
                    if (const auto *button = std::get_if<controller::GamepadButton>(&input);
                        button != nullptr && *button != controller::GamepadButton::Start)
                        assign_gamepad_fire(*button);
                    return;
                }

                if (const auto *motion = std::get_if<std::pair<GamepadAxis, double>>(&input)) {
                    const auto &[axis, value] = *motion;
                    auto &armed = axis_armed_[static_cast<size_t>(axis)];
                    if (std::abs(value) < CAPTURE_REST) armed = true;
                    else if (armed && std::abs(value) > CAPTURE_ENGAGE)
                        assign_gamepad_axis(axis, value);
                }
            }

            void unbind_conflict(const Rml::String &new_label, const Rml::String &old_label) {
                bind_warning_ = true;
                bind_status_ = new_label + " was bound to " + old_label + " — " + old_label
                               + " is now unbound.";
            }

            void assign_keyboard(const KeyboardBinding &binding) {
                const auto slots = kb_slots();
                *slots[capture_slot_] = binding;

                for (int i = 0; i < NB_KB_SLOTS; i++)
                    if (i != capture_slot_ && *slots[i] == binding) {
                        *slots[i] = std::nullopt;
                        unbind_conflict(keyboard_slot_label(binding), KB_SLOT_LABELS[i]);
                    }

                end_capture();
                rebuild_binding_rows();
            }

            void assign_gamepad_fire(const controller::GamepadButton button) {
                // single button slot: no conflict possible
                settings_.bindings.gamepad.fire = button;
                end_capture();
                rebuild_binding_rows();
            }

            void assign_gamepad_axis(const GamepadAxis axis, const double value) {
                const bool one_way = gp_slot_is_one_way(capture_slot_);
                const auto slots = gp_axis_slots();
                const GamepadAxisBinding binding{
                    .axis = axis, .sign = one_way && value < 0. ? -1.f : 1.f};
                *slots[capture_slot_ - 1] = binding;

                // two slots clash when they read the same range of an axis: a
                // two-way slot owns the whole axis, one-way slots only their
                // captured side
                for (int slot = 1; slot < NB_GP_SLOTS; slot++) {
                    if (slot == capture_slot_) continue;
                    auto &other = *slots[slot - 1];
                    if (!other || other->axis != axis) continue;
                    if (one_way && gp_slot_is_one_way(slot) && other->sign != binding.sign)
                        continue;
                    other = std::nullopt;
                    unbind_conflict(axis_slot_label(binding, one_way), GP_SLOT_LABELS[slot]);
                }

                end_capture();
                rebuild_binding_rows();
            }

            void refresh_gamepad_list() {
                auto gamepads = window_->list_gamepads();
                const bool changed =
                    gamepads.size() != gamepads_.size()
                    || !std::equal(
                        gamepads.begin(), gamepads.end(), gamepads_.begin(),
                        [](const view::GamepadInfo &a, const view::GamepadInfo &b) {
                            return a.id == b.id && a.guid == b.guid && a.name == b.name;
                        });
                if (!changed) return;

                gamepads_ = std::move(gamepads);
                gamepad_names_.clear();
                for (const auto &pad: gamepads_) gamepad_names_.emplace_back(pad.name);

                // the preferred pad when connected, else the first one (which
                // is what the window falls back to)
                selected_gamepad_ = gamepads_.empty() ? -1 : 0;
                for (size_t i = 0; i < gamepads_.size(); i++)
                    if (!settings_.bindings.gamepad.device_guid.empty()
                        && gamepads_[i].guid == settings_.bindings.gamepad.device_guid) {
                        selected_gamepad_ = static_cast<int>(i);
                        break;
                    }
                window_->select_gamepad(
                    selected_gamepad_ >= 0 ? gamepads_[selected_gamepad_].id : -1);

                if (model_handle_) {
                    model_handle_.DirtyVariable("gamepads");
                    model_handle_.DirtyVariable("selected_gamepad");
                }
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
            Rml::ElementDocument *controls_document_ = nullptr;
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

            // controls page state
            std::vector<BindingRow> kb_rows_;
            std::vector<BindingRow> gp_rows_;
            std::vector<view::GamepadInfo> gamepads_;
            std::vector<Rml::String> gamepad_names_;
            // index into gamepads_ of the pad feeding the game, -1 when none
            // is connected
            int selected_gamepad_ = -1;
            Rml::String bind_status_;
            bool bind_warning_ = false;
            // slot being captured (-1 = idle) and which page it belongs to
            int capture_slot_ = -1;
            bool capture_keyboard_page_ = false;
            std::chrono::steady_clock::time_point capture_deadline_;
            std::array<bool, NB_GAMEPAD_AXES> axis_armed_{};
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
