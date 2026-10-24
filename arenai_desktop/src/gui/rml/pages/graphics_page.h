//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H

#include <memory>
#include <string>
#include <vector>

#include <arenai_view/window.h>

#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // The Graphics screen: display mode, shadow quality, MSAA and the two GPU
    // pickers, written straight into the shared settings.
    class GraphicsPage final : public MenuPage {
    public:
        GraphicsPage(
            MenuState &state, std::shared_ptr<view::AbstractWindow> window,
            const std::vector<std::string> &gpus);

    protected:
        void bind_controls(DataModel &model) override;

    private:
        // "Auto" rides index 0 of gpu_names_, the actual devices follow
        void select_gpu(int index, int &selected, std::string &setting, const char *variable);

        GameSettings &settings_;
        std::shared_ptr<view::AbstractWindow> window_;

        Rml::String display_display_;
        Rml::String shadow_display_;
        std::vector<Rml::String> gpu_names_;
        int selected_window_gpu_ = 0;
        int selected_vision_gpu_ = 0;
        bool window_gpu_env_override_ = false;
        bool vision_gpu_env_override_ = false;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_GRAPHICS_PAGE_H
