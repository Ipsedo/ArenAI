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
#include "./loading_page.h"
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

        class RmlGui final : public AbstractGui, public AbstractOverlayRenderer {
        public:
            RmlGui(
                const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
                const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
                GuiOptions options)
                : backend_(backend), window_(backend->get_window()),
                  state_{.settings = std::move(options.initial_settings)},
                  width_(options.window_width), height_(options.window_height),
                  file_interface_(asset_reader), main_menu_(*this), loading_(*this),
                  popups_(*this) {
                Rml::SetSystemInterface(&system_interface_);
                Rml::SetFileInterface(&file_interface_);
                Rml::SetRenderInterface(&backend_->ui_render_interface());
                Rml::Initialise();

                decorators_ = std::make_unique<DecoratorRegistry>();
                register_menu_decorators(*decorators_);
                font_buffers_ = load_menu_fonts(asset_reader);

                context_ = Rml::CreateContext("menu", Rml::Vector2i(width_, height_));
                if (!context_) throw std::runtime_error("RmlUi context creation failed");
                update_dp_ratio();

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
                victory_ = std::make_unique<PopupPage>(PopupKind::Victory, state_);
                victory_->load(*context_);

                hud_ = std::make_unique<HudOverlay>(*this);
                hud_->load(*context_);

                loading_page_ = std::make_unique<LoadingPage>();
                loading_page_->load(*context_);

                select_configured_gamepad(*window_, state_.settings);
            }

            AbstractMainMenu &main_menu() override { return main_menu_; }
            AbstractLoadingScreen &loading() override { return loading_; }
            AbstractPopupMenu &popups() override { return popups_; }
            AbstractHudOverlay &hud() override { return *hud_; }

            void on_window_resized(const int width, const int height) override {
                width_ = width;
                height_ = height;
                context_->SetDimensions(Rml::Vector2i(width_, height_));
                update_dp_ratio();
            }

            ~RmlGui() override {
                window_->set_keyboard_callback(nullptr);
                window_->set_gamepad_callback(nullptr);
                window_->set_resize_callback(nullptr);

                Rml::Shutdown();
            }

        private:
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

            class LoadingScreen final : public AbstractLoadingScreen {
            public:
                explicit LoadingScreen(RmlGui &gui) : gui_(gui) {}

                void show() override {
                    gui_.loading_page_->show(Rml::ModalFlag::None, Rml::FocusFlag::None);
                }

                void render_frame() override {
                    gui_.window_->poll_events();
                    gui_.context_->Update();

                    gui_.backend_->begin_ui_pass(view::UiPass::Clear, gui_.width_, gui_.height_);
                    gui_.context_->Render();
                    gui_.backend_->end_ui_pass();
                    gui_.backend_->present();
                }

                void hide() override { gui_.loading_page_->hide(); }

            private:
                RmlGui &gui_;
            };

            class PopupMenu final : public AbstractPopupMenu {
            public:
                explicit PopupMenu(RmlGui &gui) : gui_(gui) {}

                void open(const PopupKind kind, const MatchStatus &status, PopupListener on_action)
                    override {
                    close();
                    gui_.popup(kind).open(status, std::move(on_action));
                }

                void close() override {
                    gui_.pause_->close();
                    gui_.game_over_->close();
                    gui_.victory_->close();
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
                switch (kind) {
                    case PopupKind::Pause: return *pause_;
                    case PopupKind::Victory: return *victory_;
                    default: return *game_over_;
                }
            }

            void on_escape() {
                for (PopupPage *popup: {pause_.get(), game_over_.get(), victory_.get()})
                    if (popup->visible()) {
                        popup->on_escape();
                        return;
                    }
                navigator_->on_escape();
            }

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

            void update_dp_ratio() const {
                context_->SetDensityIndependentPixelRatio(std::max(
                    0.5f, std::min(
                              static_cast<float>(width_) / 1280.0f,
                              static_cast<float>(height_) / 720.0f)));
            }

            std::shared_ptr<view::AbstractWindowedGraphicBackend> backend_;
            std::shared_ptr<view::AbstractWindow> window_;

            MenuState state_;
            int width_;
            int height_;

            MenuSystemInterface system_interface_;
            ReaderBackedFileInterface file_interface_;

            std::unique_ptr<DecoratorRegistry> decorators_;
            std::vector<std::string> font_buffers_;

            Rml::Context *context_ = nullptr;

            std::shared_ptr<MenuInputAdapter> input_adapter_;
            std::unique_ptr<Navigator> navigator_;
            std::unique_ptr<PopupPage> pause_;
            std::unique_ptr<PopupPage> game_over_;
            std::unique_ptr<PopupPage> victory_;
            std::unique_ptr<HudOverlay> hud_;
            std::unique_ptr<LoadingPage> loading_page_;

            MainMenu main_menu_;
            LoadingScreen loading_;
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
