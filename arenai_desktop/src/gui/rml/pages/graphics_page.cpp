#include "./graphics_page.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace arenai::desktop::gui {

    namespace {

        constexpr std::array SHADOW_LEVELS{
            ShadowQuality::Off, ShadowQuality::Low, ShadowQuality::Medium, ShadowQuality::High,
            ShadowQuality::Ultra};
        constexpr std::array<const char *, 5> SHADOW_NAMES{"Off", "Low", "Medium", "High", "Ultra"};
        constexpr std::array MSAA_LEVELS{1, 2, 4, 8};
        constexpr std::array<const char *, 4> MSAA_NAMES{"Off", "2x", "4x", "8x"};

        Rml::String pixels_label(const int width, const int height) {
            return std::to_string(width) + " × " + std::to_string(height);
        }

        Rml::String percent(const float value) {
            char text[16];
            std::snprintf(text, sizeof(text), "%.1f%%", value);
            return text;
        }

        // a dark calibration level as the composite's brightness curve would display it
        Rml::String calibration_color(const int level, const int brightness) {
            const float exponent = std::exp2(-0.5f * static_cast<float>(brightness) / 10.f);
            const int shown = static_cast<int>(
                std::lround(255.f * std::pow(static_cast<float>(level) / 255.f, exponent)));
            char text[16];
            std::snprintf(text, sizeof(text), "#%02X%02X%02X", shown, shown, shown);
            return text;
        }

    }// namespace

    GraphicsPage::GraphicsPage(
        MenuState &state, std::shared_ptr<view::AbstractWindow> window,
        std::function<void(bool)> apply_vsync, const std::vector<std::string> &gpus)
        : MenuPage(PageId::Graphics, PageId::Parameters, "menu/graphics.rml", "graphics"),
          settings_(state.settings), window_(std::move(window)),
          apply_vsync_(std::move(apply_vsync)),
          click_outside_listener_([this] { close_dropdowns(); }),
          display_display_(settings_.fullscreen ? "fullscreen" : "windowed"),
          native_mode_{.width = 0, .height = 0, .refresh_rate = 0} {
        const auto fullscreen_modes = window_->fullscreen_modes();
        modes_ = fullscreen_modes.modes;
        native_mode_ = fullscreen_modes.native;
        monitor_name_ = fullscreen_modes.monitor;

        // a saved mode that disappeared falls back to native, re-saved as such
        settings_.fullscreen_mode =
            available_fullscreen_mode(fullscreen_modes, settings_.fullscreen_mode);
        const view::VideoMode current = settings_.fullscreen_mode.value_or(native_mode_);
        for (size_t i = 0; i < modes_.size(); i++) {
            const auto &mode = modes_[i];
            const Rml::String rate = std::to_string(mode.refresh_rate) + " Hz";
            resolutions_.push_back(
                {.label = pixels_label(mode.width, mode.height),
                 .tag = mode == native_mode_ ? "native · " + rate : rate});
            if (mode == current) selected_resolution_ = static_cast<int>(i);
        }
        if (!resolutions_.empty()) {
            resolution_display_ = resolutions_[static_cast<size_t>(selected_resolution_)].label;
            resolution_tag_display_ = resolutions_[static_cast<size_t>(selected_resolution_)].tag;
        } else {
            resolution_display_ = pixels_label(native_mode_.width, native_mode_.height);
        }

        gpu_names_.emplace_back("Auto");
        for (const auto &gpu: gpus) gpu_names_.emplace_back(gpu);

        const auto gpu_index = [this](const std::string &name) -> int {
            for (size_t i = 1; i < gpu_names_.size(); i++)
                if (!name.empty() && gpu_names_[i] == name) return static_cast<int>(i);
            return 0;
        };
        selected_window_gpu_ = gpu_index(settings_.window_gpu);
        selected_vision_gpu_ = gpu_index(settings_.vision_gpu);
        // a saved GPU that disappeared falls back to Auto, re-saved as such
        if (selected_window_gpu_ == 0) settings_.window_gpu.clear();
        if (selected_vision_gpu_ == 0) settings_.vision_gpu.clear();
        window_gpu_display_ = gpu_names_[static_cast<size_t>(selected_window_gpu_)];
        vision_gpu_display_ = gpu_names_[static_cast<size_t>(selected_vision_gpu_)];

        window_gpu_env_override_ = std::getenv("ARENAI_VK_DEVICE_WINDOW") != nullptr;
        vision_gpu_env_override_ = std::getenv("ARENAI_VK_DEVICE") != nullptr;

        compute_brightness_labels();

        shadow_level_ = static_cast<int>(
            std::ranges::find(SHADOW_LEVELS, settings_.shadow_quality) - SHADOW_LEVELS.begin());
        const auto samples = std::ranges::find(MSAA_LEVELS, settings_.msaa_samples);
        msaa_level_ =
            samples == MSAA_LEVELS.end() ? 2 : static_cast<int>(samples - MSAA_LEVELS.begin());
        apply_quality_levels();
    }

    void GraphicsPage::bind_controls(DataModel &model) {
        if (auto entry_handle = model.register_struct<ResolutionEntry>()) {
            entry_handle.RegisterMember("label", &ResolutionEntry::label);
            entry_handle.RegisterMember("tag", &ResolutionEntry::tag);
        }
        model.register_array<std::vector<ResolutionEntry>>();
        model.register_array<std::vector<Rml::String>>();

        model.bind("display", &display_display_);
        model.bind("vsync", &settings_.vsync);
        model.bind("resolutions", &resolutions_);
        model.bind("selected_resolution", &selected_resolution_);
        model.bind("resolution", &resolution_display_);
        model.bind("resolution_tag", &resolution_tag_display_);
        model.bind("monitor", &monitor_name_);
        model.bind("resolution_open", &resolution_open_);
        model.bind("brightness", &settings_.brightness);
        model.bind("brightness_label", &brightness_display_);
        model.bind("brightness_fill_left", &brightness_fill_left_);
        model.bind("brightness_fill_width", &brightness_fill_width_);
        model.bind("calibration_visible", &calibration_visible_color_);
        model.bind("calibration_faint", &calibration_faint_color_);

        model.bind("render_scale", &settings_.render_scale);
        model.bind("render_pixels", &render_pixels_);
        model.bind("shadow_level", &shadow_level_);
        model.bind("shadow_value", &shadow_value_);
        model.bind("shadow_detail", &shadow_detail_);
        model.bind("msaa_level", &msaa_level_);
        model.bind("msaa_value", &msaa_value_);
        model.bind("ambient_occlusion", &settings_.ambient_occlusion);
        model.bind("bloom", &settings_.bloom);
        model.bind("sun_rays", &settings_.sun_rays);

        model.bind("gpus", &gpu_names_);
        model.bind("selected_window_gpu", &selected_window_gpu_);
        model.bind("selected_vision_gpu", &selected_vision_gpu_);
        model.bind("window_gpu", &window_gpu_display_);
        model.bind("vision_gpu", &vision_gpu_display_);
        model.bind("window_gpu_open", &window_gpu_open_);
        model.bind("vision_gpu_open", &vision_gpu_open_);
        model.bind("window_gpu_env", &window_gpu_env_override_);
        model.bind("vision_gpu_env", &vision_gpu_env_override_);

        model.on_event("set_display", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            display_display_ = arguments[0].Get<Rml::String>();
            settings_.fullscreen = display_display_ == "fullscreen";
            window_->set_fullscreen(settings_.fullscreen);
            if (!settings_.fullscreen) close_dropdowns();
            this->model().dirty("display");
            sync_render_scale();
        });
        model.on_event("set_vsync", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            settings_.vsync = arguments[0].Get<Rml::String>() == "on";
            apply_vsync_(settings_.vsync);
            this->model().dirty("vsync");
        });
        model.on_event("toggle_resolution", [this](const Rml::VariantList &) {
            if (!settings_.fullscreen || resolutions_.empty()) return;
            const bool open = !resolution_open_;
            close_dropdowns();
            resolution_open_ = open;
            dirty_dropdowns();
        });
        model.on_event("set_resolution", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            select_resolution(arguments[0].Get<int>());
        });
        model.on_event("sync_brightness", [this](const Rml::VariantList &arguments) {
            if (!arguments.empty()) settings_.brightness = arguments[0].Get<int>();
            sync_brightness();
        });

        model.on_event("sync_render_scale", [this](const Rml::VariantList &arguments) {
            if (!arguments.empty()) settings_.render_scale = arguments[0].Get<int>();
            sync_render_scale();
        });
        // from the slider (ev.value) or a clicked stop name
        model.on_event("set_shadow_level", [this](const Rml::VariantList &arguments) {
            if (!arguments.empty()) shadow_level_ = arguments[0].Get<int>();
            apply_quality_levels();
            dirty_quality_levels();
        });
        model.on_event("set_msaa_level", [this](const Rml::VariantList &arguments) {
            if (!arguments.empty()) msaa_level_ = arguments[0].Get<int>();
            apply_quality_levels();
            dirty_quality_levels();
        });
        model.on_event("toggle_effect", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const auto effect = arguments[0].Get<Rml::String>();
            bool *flag = effect == "ambient_occlusion" ? &settings_.ambient_occlusion
                         : effect == "bloom"           ? &settings_.bloom
                         : effect == "sun_rays"        ? &settings_.sun_rays
                                                       : nullptr;
            if (!flag) return;
            *flag = !*flag;
            this->model().dirty(effect);
        });

        model.on_event("toggle_window_gpu", [this](const Rml::VariantList &) {
            const bool open = !window_gpu_open_;
            close_dropdowns();
            window_gpu_open_ = open;
            dirty_dropdowns();
        });
        model.on_event("toggle_vision_gpu", [this](const Rml::VariantList &) {
            const bool open = !vision_gpu_open_;
            close_dropdowns();
            vision_gpu_open_ = open;
            dirty_dropdowns();
        });
        model.on_event("set_window_gpu", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            select_gpu(
                arguments[0].Get<int>(), selected_window_gpu_, settings_.window_gpu,
                "selected_window_gpu");
        });
        model.on_event("set_vision_gpu", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            select_gpu(
                arguments[0].Get<int>(), selected_vision_gpu_, settings_.vision_gpu,
                "selected_vision_gpu");
        });
    }

    void GraphicsPage::on_attached(Rml::ElementDocument &document) {
        document.AddEventListener(Rml::EventId::Click, &click_outside_listener_);
    }

    void GraphicsPage::on_show() {
        const Rml::Vector2i dimensions = document().GetContext()->GetDimensions();
        window_width_ = dimensions.x;
        window_height_ = dimensions.y;
        sync_render_scale();
    }

    bool GraphicsPage::on_back() { return close_dropdowns(); }

    void GraphicsPage::select_gpu(
        const int index, int &selected, std::string &setting, const char *variable) {
        if (index < 0 || index >= static_cast<int>(gpu_names_.size())) return;
        selected = index;
        setting = index == 0 ? std::string() : std::string(gpu_names_[index]);
        window_gpu_display_ = gpu_names_[static_cast<size_t>(selected_window_gpu_)];
        vision_gpu_display_ = gpu_names_[static_cast<size_t>(selected_vision_gpu_)];
        close_dropdowns();
        model().dirty(variable);
        model().dirty({"window_gpu", "vision_gpu"});
    }

    void GraphicsPage::select_resolution(const int index) {
        if (index < 0 || index >= static_cast<int>(modes_.size())) return;
        selected_resolution_ = index;
        const auto &mode = modes_[static_cast<size_t>(index)];
        settings_.fullscreen_mode =
            mode == native_mode_ ? std::nullopt : std::optional<view::VideoMode>(mode);
        window_->set_fullscreen_mode(settings_.fullscreen_mode);
        resolution_display_ = resolutions_[static_cast<size_t>(index)].label;
        resolution_tag_display_ = resolutions_[static_cast<size_t>(index)].tag;
        close_dropdowns();
        model().dirty({"selected_resolution", "resolution", "resolution_tag"});
        sync_render_scale();
    }

    bool GraphicsPage::close_dropdowns() {
        if (!resolution_open_ && !window_gpu_open_ && !vision_gpu_open_) return false;
        resolution_open_ = false;
        window_gpu_open_ = false;
        vision_gpu_open_ = false;
        dirty_dropdowns();
        return true;
    }

    void GraphicsPage::dirty_dropdowns() {
        model().dirty({"resolution_open", "window_gpu_open", "vision_gpu_open"});
    }

    void GraphicsPage::apply_quality_levels() {
        shadow_level_ = std::clamp(shadow_level_, 0, static_cast<int>(SHADOW_LEVELS.size()) - 1);
        msaa_level_ = std::clamp(msaa_level_, 0, static_cast<int>(MSAA_LEVELS.size()) - 1);

        const auto quality = SHADOW_LEVELS[static_cast<size_t>(shadow_level_)];
        settings_.shadow_quality = quality;
        settings_.msaa_samples = MSAA_LEVELS[static_cast<size_t>(msaa_level_)];

        shadow_value_ = SHADOW_NAMES[static_cast<size_t>(shadow_level_)];
        shadow_detail_ = quality == ShadowQuality::Off
                             ? Rml::String()
                             : std::to_string(shadow_map_size(quality)) + " px";
        msaa_value_ = MSAA_NAMES[static_cast<size_t>(msaa_level_)];
    }

    void GraphicsPage::dirty_quality_levels() {
        model().dirty(
            {"shadow_level", "shadow_value", "shadow_detail", "msaa_level", "msaa_value"});
    }

    void GraphicsPage::sync_render_scale() {
        const view::VideoMode fullscreen = settings_.fullscreen_mode.value_or(native_mode_);
        const int width = settings_.fullscreen ? fullscreen.width : window_width_;
        const int height = settings_.fullscreen ? fullscreen.height : window_height_;
        render_pixels_ = pixels_label(
            width * settings_.render_scale / 100, height * settings_.render_scale / 100);
        model().dirty("render_pixels");
    }

    void GraphicsPage::sync_brightness() {
        compute_brightness_labels();
        model().dirty(
            {"brightness_label", "brightness_fill_left", "brightness_fill_width",
             "calibration_visible", "calibration_faint"});
    }

    void GraphicsPage::compute_brightness_labels() {
        const int value = settings_.brightness;
        brightness_display_ = value > 0 ? "+" + std::to_string(value) : std::to_string(value);

        const float position = static_cast<float>(value + 10) * 5.f;
        brightness_fill_left_ = percent(std::min(50.f, position));
        brightness_fill_width_ = percent(std::abs(position - 50.f));

        calibration_visible_color_ = calibration_color(24, value);
        calibration_faint_color_ = calibration_color(10, value);
    }

}// namespace arenai::desktop::gui
