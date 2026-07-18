//
// Created by samuel on 17/07/2026.
//

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_controller/callback.h>
#include <arenai_view/window.h>

#include "./menu.h"

namespace arenai::desktop::gui {

    namespace {

        class MenuSystemInterface final : public Rml::SystemInterface {
        public:
            double GetElapsedTime() override {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
                    .count();
            }

        private:
            std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
        };

        // Serves RmlUi (documents, stylesheets, fonts) through the project's
        // asset port: whole files are pulled once via read_text — binary-safe
        // here — and the seek/read API is answered from the in-memory buffer.
        class ReaderBackedFileInterface final : public Rml::FileInterface {
        public:
            explicit ReaderBackedFileInterface(
                std::shared_ptr<utils::AbstractResourceFileReader> reader)
                : reader_(std::move(reader)) {}

            Rml::FileHandle Open(const Rml::String &path) override {
                try {
                    auto file = std::make_unique<OpenedFile>(OpenedFile{reader_->read_text(path)});
                    return reinterpret_cast<Rml::FileHandle>(file.release());
                } catch (const std::exception &e) {
                    std::cerr << "RmlUi asset open failed: " << e.what() << std::endl;
                    return 0;
                }
            }

            void Close(const Rml::FileHandle file) override {
                delete reinterpret_cast<OpenedFile *>(file);
            }

            size_t Read(void *buffer, const size_t size, const Rml::FileHandle file) override {
                auto *opened = reinterpret_cast<OpenedFile *>(file);
                const size_t nb_read = std::min(size, opened->content.size() - opened->position);
                std::memcpy(buffer, opened->content.data() + opened->position, nb_read);
                opened->position += nb_read;
                return nb_read;
            }

            bool Seek(const Rml::FileHandle file, const long offset, const int origin) override {
                auto *opened = reinterpret_cast<OpenedFile *>(file);
                const long base = origin == SEEK_CUR   ? static_cast<long>(opened->position)
                                  : origin == SEEK_END ? static_cast<long>(opened->content.size())
                                                       : 0L;
                const long target = base + offset;
                if (target < 0 || target > static_cast<long>(opened->content.size())) return false;
                opened->position = static_cast<size_t>(target);
                return true;
            }

            size_t Tell(const Rml::FileHandle file) override {
                return reinterpret_cast<OpenedFile *>(file)->position;
            }

        private:
            struct OpenedFile {
                std::string content;
                size_t position = 0;
            };

            std::shared_ptr<utils::AbstractResourceFileReader> reader_;
        };

        // The slider knob is a widget-internal element (sliderbar) that cannot
        // be wrapped in RML like the buttons, so its detached cursor ring is a
        // gui-local decorator instead: an ink ring hugging the element's edge,
        // a transparent gap, and the fill disc — plain triangle geometry, the
        // only thing the Vulkan backend renders (no box-shadow, no textures).
        // RCSS: decorator: cursor-ring(<ring-color> <fill-color> <ring-width> <gap>);
        class CursorRingDecorator final : public Rml::Decorator {
        public:
            CursorRingDecorator(
                const Rml::Colourb ring_color, const Rml::Colourb fill_color,
                const Rml::NumericValue ring_width, const Rml::NumericValue gap)
                : ring_color_(ring_color), fill_color_(fill_color), ring_width_(ring_width),
                  gap_(gap) {}

            Rml::DecoratorDataHandle
            GenerateElementData(Rml::Element *element, Rml::BoxArea) const override {
                const float opacity = element->GetComputedValues().opacity();
                const Rml::Vector2f size = element->GetBox().GetSize(Rml::BoxArea::Border);
                const float ring_width = element->ResolveLength(ring_width_);
                const float inset = ring_width + element->ResolveLength(gap_);
                // ring and fill both adopt the element's border-radius: the
                // ring must read as the same shape as the knob it surrounds
                // (a concentric radius+inset ring looks round next to a
                // square knob), so no radius inflation on the outer box
                const float ring_radius = element->GetComputedValues().border_top_left_radius();
                const float fill_radius = ring_radius;

                Rml::Mesh mesh;

                // hollow ring at the outer edge (zero-alpha background: only
                // the border geometry is emitted)
                const Rml::RenderBox ring_box(
                    {size.x - 2.f * ring_width, size.y - 2.f * ring_width}, {0.f, 0.f},
                    {ring_width, ring_width, ring_width, ring_width},
                    {ring_radius, ring_radius, ring_radius, ring_radius});
                const Rml::ColourbPremultiplied ring = ring_color_.ToPremultiplied(opacity);
                const Rml::ColourbPremultiplied ring_colors[4] = {ring, ring, ring, ring};
                Rml::MeshUtilities::GenerateBackgroundBorder(
                    mesh, ring_box, Rml::ColourbPremultiplied(0, 0, 0, 0), ring_colors);

                // fill disc, inset past the ring and the gap
                const Rml::Vector2f fill_size = {size.x - 2.f * inset, size.y - 2.f * inset};
                const Rml::RenderBox fill_box(
                    fill_size, {inset, inset}, {0.f, 0.f, 0.f, 0.f},
                    {fill_radius, fill_radius, fill_radius, fill_radius});
                Rml::MeshUtilities::GenerateBackground(
                    mesh, fill_box, fill_color_.ToPremultiplied(opacity));

                auto *geometry =
                    new Rml::Geometry(element->GetRenderManager()->MakeGeometry(std::move(mesh)));
                return reinterpret_cast<Rml::DecoratorDataHandle>(geometry);
            }

            void ReleaseElementData(const Rml::DecoratorDataHandle element_data) const override {
                delete reinterpret_cast<Rml::Geometry *>(element_data);
            }

            void RenderElement(
                Rml::Element *element, const Rml::DecoratorDataHandle element_data) const override {
                reinterpret_cast<Rml::Geometry *>(element_data)
                    ->Render(element->GetAbsoluteOffset(Rml::BoxArea::Border));
            }

        private:
            Rml::Colourb ring_color_;
            Rml::Colourb fill_color_;
            Rml::NumericValue ring_width_;
            Rml::NumericValue gap_;
        };

        class CursorRingDecoratorInstancer final : public Rml::DecoratorInstancer {
        public:
            CursorRingDecoratorInstancer() {
                id_ring_color_ =
                    RegisterProperty("ring-color", "#EAF7FF").AddParser("color").GetId();
                id_fill_color_ =
                    RegisterProperty("fill-color", "#00A6FB").AddParser("color").GetId();
                id_ring_width_ = RegisterProperty("ring-width", "2dp").AddParser("length").GetId();
                id_gap_ = RegisterProperty("gap", "2dp").AddParser("length").GetId();
                RegisterShorthand(
                    "decorator", "ring-color, fill-color, ring-width, gap",
                    Rml::ShorthandType::FallThrough);
            }

            Rml::SharedPtr<Rml::Decorator> InstanceDecorator(
                const Rml::String &, const Rml::PropertyDictionary &properties,
                const Rml::DecoratorInstancerInterface &) override {
                return Rml::MakeShared<CursorRingDecorator>(
                    properties.GetProperty(id_ring_color_)->Get<Rml::Colourb>(),
                    properties.GetProperty(id_fill_color_)->Get<Rml::Colourb>(),
                    properties.GetProperty(id_ring_width_)->GetNumericValue(),
                    properties.GetProperty(id_gap_)->GetNumericValue());
            }

        private:
            Rml::PropertyId id_ring_color_{};
            Rml::PropertyId id_fill_color_{};
            Rml::PropertyId id_ring_width_{};
            Rml::PropertyId id_gap_{};
        };

        // Adapts the window's input port to RmlUi context events: the menus
        // reuse the same controller callbacks as the game, so GLFW types
        // never surface here. The gamepad half rides RmlUi's keyboard
        // navigation: A activates the focused element (Enter), B backs out
        // (the Escape path), D-pad and left stick move the spatial focus.
        //
        // Only one pointing device highlights the menu at a time — whichever
        // spoke last. A gamepad action clears the mouse hover and flags the
        // documents (.gamepad-nav, which menu.rcss needs to show the :focus
        // highlight); any mouse activity lifts the flag, and hover takes over
        // again on the next cursor move. The RmlUi focus itself is kept across
        // switches so the gamepad resumes right where it left the menu.
        class MenuInputAdapter final : public controller::AbstractKeyboardCallback,
                                       public controller::AbstractGamepadCallback {
        public:
            MenuInputAdapter(
                Rml::Context *context, std::function<void()> on_escape,
                std::function<void(bool)> on_gamepad_nav)
                : context_(context), on_escape_(std::move(on_escape)),
                  on_gamepad_nav_(std::move(on_gamepad_nav)) {}

            void on_key(const controller::Key key, const controller::InputAction action) override {
                if (key == controller::Key::Escape && action == controller::InputAction::Press)
                    on_escape_();
            }

            void on_mouse_move(const double x, const double y) override {
                to_mouse_mode();
                context_->ProcessMouseMove(static_cast<int>(x), static_cast<int>(y), 0);
            }

            void on_mouse_button(
                const controller::MouseButton button,
                const controller::InputAction action) override {
                to_mouse_mode();
                const int index = button == controller::MouseButton::Left    ? 0
                                  : button == controller::MouseButton::Right ? 1
                                                                             : 2;
                if (action == controller::InputAction::Press)
                    context_->ProcessMouseButtonDown(index, 0);
                else if (action == controller::InputAction::Release)
                    context_->ProcessMouseButtonUp(index, 0);
            }

            void on_scroll(const double x_offset, const double y_offset) override {
                to_mouse_mode();
                // GLFW wheel offsets are positive upwards, RmlUi scrolls
                // positive downwards
                context_->ProcessMouseWheel(
                    Rml::Vector2f(-static_cast<float>(x_offset), -static_cast<float>(y_offset)), 0);
            }

            void on_gamepad_button(
                const controller::GamepadButton button,
                const controller::InputAction action) override {
                if (action != controller::InputAction::Press) return;

                to_gamepad_mode();
                switch (button) {
                    case controller::GamepadButton::A: send_key(Rml::Input::KI_RETURN); break;
                    case controller::GamepadButton::B: on_escape_(); break;
                    case controller::GamepadButton::DPadUp: send_key(Rml::Input::KI_UP); break;
                    case controller::GamepadButton::DPadDown: send_key(Rml::Input::KI_DOWN); break;
                    case controller::GamepadButton::DPadLeft: send_key(Rml::Input::KI_LEFT); break;
                    case controller::GamepadButton::DPadRight:
                        send_key(Rml::Input::KI_RIGHT);
                        break;
                    default: break;
                }
            }

            void on_joystick(
                const double x, const double y, const controller::GamepadJoystick stick) override {
                if (stick != controller::GamepadJoystick::Left) return;

                // GLFW stick y grows downwards, matching KI_DOWN
                axis_nav(x_nav_, x, Rml::Input::KI_LEFT, Rml::Input::KI_RIGHT);
                axis_nav(y_nav_, y, Rml::Input::KI_UP, Rml::Input::KI_DOWN);
            }

            // triggers have no menu role
            void on_trigger(double, controller::GamepadTrigger) override {}

        private:
            void to_gamepad_mode() {
                if (gamepad_mode_) return;
                gamepad_mode_ = true;
                // the hover chain would otherwise stay highlighted under the
                // idle cursor; the next ProcessMouseMove reactivates it
                context_->ProcessMouseLeave();
                on_gamepad_nav_(true);
            }

            void to_mouse_mode() {
                if (!gamepad_mode_) return;
                gamepad_mode_ = false;
                on_gamepad_nav_(false);
            }

            // one navigation step per key: RmlUi acts on the down event
            void send_key(const Rml::Input::KeyIdentifier key) {
                to_gamepad_mode();
                context_->ProcessKeyDown(key, 0);
                context_->ProcessKeyUp(key, 0);
            }

            struct AxisNav {
                int direction = 0;// -1 / 0 / +1 after hysteresis
                std::chrono::steady_clock::time_point next_repeat;
            };

            // Hysteresis (engage past 0.5, release under 0.3) turns the analog
            // deflection into clean steps, and holding the stick auto-repeats
            // like a held keyboard arrow; the window dispatches the stick once
            // per frame, which paces the repeat clock.
            void axis_nav(
                AxisNav &nav, const double value, const Rml::Input::KeyIdentifier negative_key,
                const Rml::Input::KeyIdentifier positive_key) {
                constexpr double ENGAGE = 0.5, RELEASE = 0.3;
                constexpr auto FIRST_REPEAT = std::chrono::milliseconds(400);
                constexpr auto NEXT_REPEAT = std::chrono::milliseconds(120);

                const int direction = value <= -ENGAGE ? -1 : value >= ENGAGE ? 1 : 0;
                const bool held = nav.direction != 0 && value * nav.direction > RELEASE;
                const auto now = std::chrono::steady_clock::now();

                if (direction != 0 && direction != nav.direction) {
                    nav.direction = direction;
                    nav.next_repeat = now + FIRST_REPEAT;
                    send_key(direction < 0 ? negative_key : positive_key);
                } else if (held && now >= nav.next_repeat) {
                    nav.next_repeat = now + NEXT_REPEAT;
                    send_key(nav.direction < 0 ? negative_key : positive_key);
                } else if (!held && direction == 0) {
                    nav.direction = 0;
                }
            }

            Rml::Context *context_;
            std::function<void()> on_escape_;
            std::function<void(bool)> on_gamepad_nav_;
            bool gamepad_mode_ = false;
            AxisNav x_nav_;
            AxisNav y_nav_;
        };

        class RmlGui final : public AbstractGui {
        public:
            RmlGui(
                const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
                const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
                const GameSettings &initial_settings, const int window_width,
                const int window_height)
                : backend_(backend), window_(backend->get_window()), settings_(initial_settings),
                  width_(window_width), height_(window_height), file_interface_(asset_reader),
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

                load_fonts(asset_reader);

                context_ = Rml::CreateContext("menu", Rml::Vector2i(width_, height_));
                if (!context_) throw std::runtime_error("RmlUi context creation failed");
                update_dp_ratio();

                current_dir_ = std::filesystem::exists(settings_.sac_folder)
                                   ? std::filesystem::canonical(settings_.sac_folder)
                                   : std::filesystem::current_path();

                bind_data_model();
                refresh_explorer();

                main_document_ = context_->LoadDocument("menu/main_menu.rml");
                params_document_ = context_->LoadDocument("menu/parameters.rml");
                pause_document_ = context_->LoadDocument("menu/pause.rml");
                if (!main_document_ || !params_document_ || !pause_document_)
                    throw std::runtime_error("RmlUi menu documents failed to load");

                input_adapter_ = std::make_shared<MenuInputAdapter>(
                    context_,
                    [this] {
                        // Escape / gamepad B back out of the parameters screen;
                        // while paused B resumes the game (the application
                        // intercepts Escape itself before this adapter)
                        if (pause_document_->IsVisible())
                            pending_pause_action_ = PauseAction::Continue;
                        else if (params_document_->IsVisible()) close_params();
                    },
                    [this](const bool gamepad) {
                        // menu.rcss shows the :focus highlight only under
                        // .gamepad-nav, so the mouse hover and the gamepad
                        // cursor are never visible together
                        for (auto *document: {main_document_, params_document_, pause_document_})
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

            void open_pause() override {
                pending_pause_action_ = PauseAction::None;
                pause_document_->Show();
            }

            void close_pause() override { pause_document_->Hide(); }

            void render_pause_overlay() override {
                context_->Update();

                backend_->begin_ui_overlay(width_, height_);
                context_->Render();
                backend_->end_ui_frame();
            }

            PauseAction poll_pause_action() override {
                return std::exchange(pending_pause_action_, PauseAction::None);
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
                context_->SetDensityIndependentPixelRatio(
                    std::max(0.5f, std::min(screen_width / 1920.0f, screen_height / 1080.0f)));
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
                    {"font/Sora-Regular.ttf", "Sora", 400},
                    {"font/Sora-SemiBold.ttf", "Sora", 600},
                    {"font/Sora-Bold.ttf", "Sora", 700},
                    {"font/IBMPlexMono-Regular.ttf", "IBM Plex Mono", 400},
                    {"font/IBMPlexMono-Medium.ttf", "IBM Plex Mono", 500},
                    {"font/IBMPlexMono-SemiBold.ttf", "IBM Plex Mono", 600},
                };

                font_buffers_.reserve(std::size(MENU_FONTS));
                for (const auto &[path, family, weight]: MENU_FONTS) {
                    font_buffers_.push_back(asset_reader->read_text(path));
                    const auto &buffer = font_buffers_.back();
                    Rml::LoadFontFace(
                        Rml::Span<const Rml::byte>(
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
                constructor.Bind("current_dir", &current_dir_display_);
                constructor.Bind("entries", &entries_);
                constructor.Bind("can_play", &can_play_);

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
                can_play_ = !settings_.sac_folder.empty()
                            && std::filesystem::is_directory(settings_.sac_folder);

                if (model_handle_) {
                    model_handle_.DirtyVariable("entries");
                    model_handle_.DirtyVariable("current_dir");
                    model_handle_.DirtyVariable("sac_folder");
                    model_handle_.DirtyVariable("can_play");
                }
            }

            void close_params() {
                params_document_->Hide();
                main_document_->Show();
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
            std::vector<std::string> font_buffers_;

            Rml::Context *context_ = nullptr;
            Rml::ElementDocument *main_document_ = nullptr;
            Rml::ElementDocument *params_document_ = nullptr;
            Rml::ElementDocument *pause_document_ = nullptr;
            Rml::DataModelHandle model_handle_;

            std::shared_ptr<MenuInputAdapter> input_adapter_;

            std::filesystem::path current_dir_;
            Rml::String controller_display_;
            Rml::String display_display_;
            Rml::String current_dir_display_;
            Rml::String sac_folder_display_;
            std::vector<Rml::String> entries_;
            bool can_play_ = false;
            bool play_clicked_ = false;
            bool quit_clicked_ = false;
            PauseAction pending_pause_action_ = PauseAction::None;
        };

    }// namespace

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
        const GameSettings &initial_settings, const int window_width, const int window_height) {
        return std::make_unique<RmlGui>(
            backend, asset_reader, initial_settings, window_width, window_height);
    }

}// namespace arenai::desktop::gui
