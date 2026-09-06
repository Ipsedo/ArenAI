//
// Created by samuel on 17/07/2026.
//

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_controller/callback.h>
#include <arenai_view/window.h>

#include "../menu.h"
#include "./adapters.h"
#include "./cursor_ring.h"
#include "./damage_arc.h"
#include "./hit_marker.h"
#include "./hud.h"
#include "./input.h"
#include "./pages/ai_page.h"
#include "./pages/controls_page.h"
#include "./pages/graphics_page.h"
#include "./reticle.h"

namespace arenai::desktop::gui {

    namespace {

        // Orchestrates the RmlUi machinery: library lifecycle, the loaded
        // documents and the navigation between them, the main-menu loop and
        // the pause / game-over popups. Everything page-specific lives in the
        // pages/ classes, the in-game overlay in Hud.
        class RmlGui final : public AbstractGui {
        public:
            RmlGui(
                const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
                const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
                const GameSettings &initial_settings, const std::vector<std::string> &gpus,
                const int window_width, const int window_height, AgentValidator agent_validator)
                : backend_(backend), window_(backend->get_window()), settings_(initial_settings),
                  width_(window_width), height_(window_height), file_interface_(asset_reader),
                  graphics_page_(settings_, window_, gpus), controls_page_(settings_, window_),
                  ai_page_(settings_, std::move(agent_validator)) {
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
                damage_arc_instancer_ = std::make_unique<DamageArcDecoratorInstancer>();
                Rml::Factory::RegisterDecoratorInstancer("damage-arc", damage_arc_instancer_.get());

                load_fonts(asset_reader);

                context_ = Rml::CreateContext("menu", Rml::Vector2i(width_, height_));
                if (!context_) throw std::runtime_error("RmlUi context creation failed");
                update_dp_ratio();

                bind_data_model();
                ai_page_.refresh_explorer();

                main_document_ = context_->LoadDocument("menu/main_menu.rml");
                params_document_ = context_->LoadDocument("menu/parameters.rml");
                controls_document_ = context_->LoadDocument("menu/controls.rml");
                graphics_document_ = context_->LoadDocument("menu/graphics.rml");
                ai_document_ = context_->LoadDocument("menu/ai.rml");
                pause_document_ = context_->LoadDocument("menu/pause.rml");
                game_over_document_ = context_->LoadDocument("menu/game_over.rml");
                hud_document_ = context_->LoadDocument("menu/hud.rml");
                if (!main_document_ || !params_document_ || !controls_document_
                    || !graphics_document_ || !ai_document_ || !pause_document_
                    || !game_over_document_ || !hud_document_)
                    throw std::runtime_error("RmlUi menu documents failed to load");

                hud_ = std::make_unique<Hud>(*hud_document_);
                hud_document_->Show(Rml::ModalFlag::None, Rml::FocusFlag::None);

                // D-pad bridge across the file explorer's scroll container
                ai_page_.attach(*ai_document_);

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
                        else if (graphics_document_->IsVisible()) close_graphics();
                        else if (ai_document_->IsVisible()) close_ai();
                        else if (params_document_->IsVisible()) close_params();
                    },
                    [this](const bool gamepad) {
                        // menu.rcss shows the :focus highlight only under
                        // .gamepad-nav, so the mouse hover and the gamepad
                        // cursor are never visible together
                        for (auto *document:
                             {main_document_, params_document_, controls_document_,
                              graphics_document_, ai_document_, pause_document_,
                              game_over_document_})
                            document->SetClass("gamepad-nav", gamepad);
                    });
                controls_page_.set_input_adapter(input_adapter_);

                // route the pad input to the pad persisted from the previous
                // session, when it is connected
                controls_page_.on_open();
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

                    controls_page_.update(controls_document_->IsVisible());

                    context_->Update();

                    ai_page_.update_after_context();

                    backend_->begin_ui_frame(width_, height_);
                    context_->Render();
                    backend_->end_ui_frame();
                    backend_->present();
                }

                main_document_->Hide();
                params_document_->Hide();
                graphics_document_->Hide();
                ai_document_->Hide();
                controls_page_.on_close();
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

            void notify_hit(const HitKind kind) override { hud_->notify_hit(kind); }

            void notify_damage(const float screen_angle) override {
                hud_->notify_damage(screen_angle);
            }

            void set_aim_point(const std::optional<glm::vec2> normalized) override {
                hud_->set_aim_point(normalized, width_, height_);
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
                constructor.Bind("score", &score_);

                graphics_page_.bind(constructor);
                controls_page_.bind(constructor);
                ai_page_.bind(constructor);

                constructor.BindEventCallback(
                    "play", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        if (ai_page_.can_play()) play_clicked_ = true;
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
                    "open_graphics",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        params_document_->Hide();
                        graphics_document_->Show();
                    });
                constructor.BindEventCallback(
                    "graphics_back",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        close_graphics();
                    });
                constructor.BindEventCallback(
                    "open_controls",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        params_document_->Hide();
                        controls_page_.on_open();
                        controls_document_->Show();
                    });
                constructor.BindEventCallback(
                    "controls_back",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        close_controls();
                    });
                constructor.BindEventCallback(
                    "open_ai",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        params_document_->Hide();
                        ai_document_->Show();
                    });
                constructor.BindEventCallback(
                    "ai_back",
                    [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                        close_ai();
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
                graphics_page_.set_model_handle(model_handle_);
                controls_page_.set_model_handle(model_handle_);
                ai_page_.set_model_handle(model_handle_);
            }

            void close_params() const {
                params_document_->Hide();
                main_document_->Show();
            }

            void close_ai() const {
                ai_document_->Hide();
                params_document_->Show();
            }

            void close_graphics() const {
                graphics_document_->Hide();
                params_document_->Show();
            }

            void close_controls() {
                controls_page_.on_close();
                controls_document_->Hide();
                params_document_->Show();
            }

            std::shared_ptr<view::AbstractWindowedGraphicBackend> backend_;
            std::shared_ptr<view::AbstractWindow> window_;

            GameSettings settings_;
            int width_;
            int height_;

            MenuSystemInterface system_interface_;
            ReaderBackedFileInterface file_interface_;

            // the pages register their slice of the "settings" data model and
            // write straight into settings_
            GraphicsPage graphics_page_;
            ControlsPage controls_page_;
            AiPage ai_page_;

            // unique_ptr: created after Rml::Initialise(), and member
            // destruction keeps it alive until after Rml::Shutdown() as
            // RmlUi requires of registered instancers
            std::unique_ptr<CursorRingDecoratorInstancer> cursor_ring_instancer_;
            std::unique_ptr<HitMarkerDecoratorInstancer> hit_marker_instancer_;
            std::unique_ptr<ReticleDecoratorInstancer> reticle_instancer_;
            std::unique_ptr<DamageArcDecoratorInstancer> damage_arc_instancer_;
            std::vector<std::string> font_buffers_;

            Rml::Context *context_ = nullptr;
            Rml::ElementDocument *main_document_ = nullptr;
            Rml::ElementDocument *params_document_ = nullptr;
            Rml::ElementDocument *controls_document_ = nullptr;
            Rml::ElementDocument *graphics_document_ = nullptr;
            Rml::ElementDocument *ai_document_ = nullptr;
            Rml::ElementDocument *pause_document_ = nullptr;
            Rml::ElementDocument *game_over_document_ = nullptr;
            Rml::ElementDocument *hud_document_ = nullptr;
            // built once hud_document_ is loaded
            std::unique_ptr<Hud> hud_;
            Rml::DataModelHandle model_handle_;

            std::shared_ptr<MenuInputAdapter> input_adapter_;

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
        const GameSettings &initial_settings, const std::vector<std::string> &gpus,
        const int window_width, const int window_height, AgentValidator agent_validator) {
        return std::make_unique<RmlGui>(
            backend, asset_reader, initial_settings, gpus, window_width, window_height,
            std::move(agent_validator));
    }

}// namespace arenai::desktop::gui
