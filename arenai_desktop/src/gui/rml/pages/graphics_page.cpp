//
// Created by samuel on 06/09/2026.
//

#include "./graphics_page.h"

#include <cstdlib>
#include <utility>

namespace arenai::desktop::gui {

    GraphicsPage::GraphicsPage(
        MenuState &state, std::shared_ptr<view::AbstractWindow> window,
        const std::vector<std::string> &gpus)
        : MenuPage(PageId::Graphics, PageId::Parameters, "menu/graphics.rml", "graphics"),
          settings_(state.settings), window_(std::move(window)),
          display_display_(settings_.fullscreen ? "fullscreen" : "windowed"),
          shadow_display_(to_string(settings_.shadow_quality)) {
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

        window_gpu_env_override_ = std::getenv("ARENAI_VK_DEVICE_WINDOW") != nullptr;
        vision_gpu_env_override_ = std::getenv("ARENAI_VK_DEVICE") != nullptr;
    }

    void GraphicsPage::bind_controls(DataModel &model) {
        model.register_array<std::vector<Rml::String>>();

        model.bind("display", &display_display_);
        model.bind("shadow_quality", &shadow_display_);
        model.bind("msaa", &settings_.msaa_samples);
        model.bind("gpus", &gpu_names_);
        model.bind("selected_window_gpu", &selected_window_gpu_);
        model.bind("selected_vision_gpu", &selected_vision_gpu_);
        model.bind("window_gpu_env", &window_gpu_env_override_);
        model.bind("vision_gpu_env", &vision_gpu_env_override_);

        model.on_event("set_display", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            display_display_ = arguments[0].Get<Rml::String>();
            settings_.fullscreen = display_display_ == "fullscreen";
            // applied immediately; the window reports its new size through
            // the resize callback (dp-ratio included)
            window_->set_fullscreen(settings_.fullscreen);
            this->model().dirty("display");
        });
        model.on_event("set_shadow_quality", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const auto quality = shadow_quality_from_string(arguments[0].Get<Rml::String>());
            if (!quality) return;
            settings_.shadow_quality = *quality;
            shadow_display_ = to_string(*quality);
            this->model().dirty("shadow_quality");
        });
        model.on_event("set_msaa", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            const int samples = arguments[0].Get<int>();
            if (samples != 1 && samples != 2 && samples != 4 && samples != 8) return;
            settings_.msaa_samples = samples;
            this->model().dirty("msaa");
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

    void GraphicsPage::select_gpu(
        const int index, int &selected, std::string &setting, const char *variable) {
        if (index < 0 || index >= static_cast<int>(gpu_names_.size())) return;
        selected = index;
        setting = index == 0 ? std::string() : std::string(gpu_names_[index]);
        model().dirty(variable);
    }

}// namespace arenai::desktop::gui
