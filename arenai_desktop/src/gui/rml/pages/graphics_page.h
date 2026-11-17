#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arenai_view/window.h>

#include "../click_outside_listener.h"
#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // one fullscreen mode of the resolution dropdown
    struct ResolutionEntry {
        Rml::String label;
        Rml::String tag;
    };

    class GraphicsPage final : public MenuPage {
    public:
        GraphicsPage(
            MenuState &state, std::shared_ptr<view::AbstractWindow> window,
            std::function<void(bool)> apply_vsync, const std::vector<std::string> &gpus);

        // Escape / gamepad B folds an open dropdown first
        bool on_back() override;

    protected:
        void bind_controls(DataModel &model) override;
        // hooks the click-outside closing of the dropdowns
        void on_attached(Rml::ElementDocument &document) override;
        void on_show() override;

    private:
        // "Auto" rides index 0 of gpu_names_, the actual devices follow
        void select_gpu(int index, int &selected, std::string &setting, const char *variable);
        void select_resolution(int index);
        // folds every list; true when one was open
        bool close_dropdowns();
        void dirty_dropdowns();

        void apply_quality_levels();
        void dirty_quality_levels();
        void sync_render_scale();
        void sync_brightness();
        void compute_brightness_labels();

        GameSettings &settings_;
        std::shared_ptr<view::AbstractWindow> window_;
        std::function<void(bool)> apply_vsync_;

        ClickOutsideListener click_outside_listener_;

        Rml::String display_display_;
        // slider stops: ShadowQuality order, then 1/2/4/8 samples
        int shadow_level_ = 0;
        int msaa_level_ = 0;
        Rml::String shadow_value_;
        Rml::String shadow_detail_;
        Rml::String msaa_value_;

        std::vector<view::VideoMode> modes_;
        view::VideoMode native_mode_;
        std::vector<ResolutionEntry> resolutions_;
        int selected_resolution_ = 0;
        Rml::String resolution_display_;
        Rml::String resolution_tag_display_;
        Rml::String monitor_name_;
        bool resolution_open_ = false;

        // pixels, the base the render scale applies to
        int window_width_ = 0;
        int window_height_ = 0;
        Rml::String render_pixels_;

        Rml::String brightness_display_;
        Rml::String brightness_fill_left_;
        Rml::String brightness_fill_width_;
        Rml::String calibration_visible_color_;
        Rml::String calibration_faint_color_;

        std::vector<Rml::String> gpu_names_;
        int selected_window_gpu_ = 0;
        int selected_vision_gpu_ = 0;
        // the selected names, what the folded dropdown heads display
        Rml::String window_gpu_display_;
        Rml::String vision_gpu_display_;
        bool window_gpu_open_ = false;
        bool vision_gpu_open_ = false;
        bool window_gpu_env_override_ = false;
        bool vision_gpu_env_override_ = false;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H
