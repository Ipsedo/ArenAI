//
// Created by samuel on 17/07/2026.
//

#include "../gui.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_view/window.h>

#include "../../controller/gamepad_selection.h"
#include "./adapters.h"
#include "./decorators/decorator_registry.h"
#include "./fonts.h"
#include "./hud.h"
#include "./input.h"
#include "./menu_state.h"
#include "./navigator.h"
#include "./pages/ai_page.h"
#include "./pages/controls_page.h"
#include "./pages/graphics_page.h"
#include "./pages/main_page.h"
#include "./pages/parameters_page.h"
#include "./popup_page.h"

namespace arenai::desktop::gui {

    namespace {

        // The composition root of the RmlUi implementation: library
        // lifecycle, the context, and the wiring of the pages, popups and
        // HUD. Each role of the port is served by a small object below;
        // everything page-specific lives in the pages.
        class RmlGui final : public AbstractGui, public AbstractOverlayRenderer {
        public:
            RmlGui(
                const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
                const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
                GuiOptions options)
                : backend_(backend), window_(backend->get_window()),
                  state_{.settings = std::move(options.initial_settings)},
                  width_(options.window_width), height_(options.window_height),
                  file_interface_(asset_reader), main_menu_(*this), popups_(*this) {
                Rml::SetSystemInterface(&system_interface_);
                Rml::SetFileInterface(&file_interface_);
                Rml::SetRenderInterface(&backend_->ui_render_interface());
                Rml::Initialise();

                // built only now: property registration needs the style-sheet
                // specification Rml::Initialise() just created
                decorators_ = std::make_unique<DecoratorRegistry>();
                register_menu_decorators(*decorators_);
                font_buffers_ = load_menu_fonts(asset_reader);

                context_ = Rml::CreateContext("menu", Rml::Vector2i(width_, height_));
                if (!context_) throw std::runtime_error("RmlUi context creation failed");
                update_dp_ratio();

                // the adapter is wired before the pages (the controls page
                // captures bindings through it); input only arrives after
                // construction, once the documents are loaded
                input_adapter_ = std::make_shared<MenuInputAdapter>(
                    context_, [this] { on_escape(); },
                    [this](const bool gamepad) { set_gamepad_nav(gamepad); });

                std::vector<std::unique_ptr<MenuPage>> pages;
                pages.push_back(std::make_unique<MainPage>(
                    state_, [this](const MenuOutcome outcome) { main_menu_.finish(outcome); }));
                pages.push_back(std::make_unique<ParametersPage>(state_));
                pages.push_back(std::make_unique<GraphicsPage>(state_, window_, options.gpus));
                pages.push_back(std::make_unique<ControlsPage>(state_, window_, *input_adapter_));
                pages.push_back(std::make_unique<AiPage>(
                    state_, std::move(options.presets), std::move(options.agent_validator),
                    options.cuda_available));
                navigator_ = std::make_unique<Navigator>(*context_, std::move(pages), PageId::Main);

                pause_ = std::make_unique<PopupPage>(PopupKind::Pause, state_);
                pause_->load(*context_);
                game_over_ = std::make_unique<PopupPage>(PopupKind::GameOver, state_);
                game_over_->load(*context_);

                hud_ = std::make_unique<HudOverlay>(*this);
                hud_->load(*context_);

                // route the pad input to the pad persisted from the previous
                // session, when it is connected
                select_configured_gamepad(*window_, state_.settings);
            }

            AbstractMainMenu &main_menu() override { return main_menu_; }
            AbstractPopupMenu &popups() override { return popups_; }
            AbstractHudOverlay &hud() override { return *hud_; }

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

                // releases the GPU resources through the backend's render
                // interface; the instancers (decorators_) and font buffers
                // are members, hence destroyed after this call as RmlUi needs
                Rml::Shutdown();
            }

        private:
            // the blocking main-menu loop, owning the window input meanwhile
            class MainMenu final : public AbstractMainMenu {
            public:
                explicit MainMenu(RmlGui &gui) : gui_(gui) {}

                MenuOutcome run() override {
                    outcome_.reset();

                    gui_.window_->set_keyboard_callback(gui_.input_adapter_);
                    gui_.window_->set_gamepad_callback(gui_.input_adapter_);
                    gui_.window_->set_cursor_mode(controller::CursorMode::Normal);
                    gui_.navigator_->show_root();

                    while (!gui_.window_->should_close() && !outcome_) {
                        gui_.window_->poll_events();

                        gui_.navigator_->update();
                        gui_.context_->Update();
                        gui_.navigator_->after_context_update();

                        gui_.backend_->begin_ui_pass(
                            view::UiPass::Clear, gui_.width_, gui_.height_);
                        gui_.context_->Render();
                        gui_.backend_->end_ui_pass();
                        gui_.backend_->present();
                    }

                    gui_.navigator_->hide_all();
                    gui_.window_->set_keyboard_callback(nullptr);
                    gui_.window_->set_gamepad_callback(nullptr);

                    return outcome_.value_or(MenuOutcome::Quit);
                }

                const GameSettings &settings() const override { return gui_.state_.settings; }

                void finish(const MenuOutcome outcome) { outcome_ = outcome; }

            private:
                RmlGui &gui_;
                std::optional<MenuOutcome> outcome_;
            };

            class PopupMenu final : public AbstractPopupMenu {
            public:
                explicit PopupMenu(RmlGui &gui) : gui_(gui) {}

                void open(const PopupKind kind, const int score, PopupListener on_action) override {
                    close();
                    gui_.popup(kind).open(score, std::move(on_action));
                }

                void close() override {
                    gui_.pause_->close();
                    gui_.game_over_->close();
                }

                void render() override { gui_.render_overlay(); }

                std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_input() override {
                    return gui_.input_adapter_;
                }

                std::shared_ptr<controller::AbstractGamepadCallback> gamepad_input() override {
                    return gui_.input_adapter_;
                }

            private:
                RmlGui &gui_;
            };

            PopupPage &popup(const PopupKind kind) const {
                return kind == PopupKind::Pause ? *pause_ : *game_over_;
            }

            // Escape / gamepad B: an open popup first (the application
            // intercepts Escape itself in-game, this covers the gamepad),
            // else the menu stack
            void on_escape() {
                for (PopupPage *popup: {pause_.get(), game_over_.get()})
                    if (popup->visible()) {
                        popup->on_escape();
                        return;
                    }
                navigator_->on_escape();
            }

            // the .gamepad-nav class on every document of the context — pages,
            // popups and HUD alike (menu.rcss shows the :focus highlight only
            // under it)
            void set_gamepad_nav(const bool gamepad) const {
                for (int i = 0; i < context_->GetNumDocuments(); ++i)
                    context_->GetDocument(i)->SetClass("gamepad-nav", gamepad);
            }

            void render_overlay() override {
                context_->Update();

                backend_->begin_ui_pass(view::UiPass::Overlay, width_, height_);
                context_->Render();
                backend_->end_ui_pass();
            }

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

            std::shared_ptr<view::AbstractWindowedGraphicBackend> backend_;
            std::shared_ptr<view::AbstractWindow> window_;

            MenuState state_;
            int width_;
            int height_;

            MenuSystemInterface system_interface_;
            ReaderBackedFileInterface file_interface_;

            // created after Rml::Initialise(), destroyed after Rml::Shutdown()
            std::unique_ptr<DecoratorRegistry> decorators_;
            std::vector<std::string> font_buffers_;

            Rml::Context *context_ = nullptr;

            std::shared_ptr<MenuInputAdapter> input_adapter_;
            std::unique_ptr<Navigator> navigator_;
            std::unique_ptr<PopupPage> pause_;
            std::unique_ptr<PopupPage> game_over_;
            std::unique_ptr<HudOverlay> hud_;

            MainMenu main_menu_;
            PopupMenu popups_;
        };

    }// namespace

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
        GuiOptions options) {
        return std::make_unique<RmlGui>(backend, asset_reader, std::move(options));
    }

}// namespace arenai::desktop::gui
