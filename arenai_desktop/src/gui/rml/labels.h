//
// Created by samuel on 27/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_LABELS_H
#define ARENAI_DESKTOP_GUI_RML_LABELS_H

#include <algorithm>
#include <cstdio>
#include <string>

#include "../../core/settings.h"

namespace arenai::desktop::gui {

    // Human labels of the settings enums, for the one-line summaries the
    // Main and Parameters screens show (the RML keeps the raw to_string
    // values for its data-class bindings).
    constexpr const char *algorithm_label(const AiAlgorithm algorithm) {
        switch (algorithm) {
            case AiAlgorithm::Ppo: return "PPO";
            case AiAlgorithm::PpoLiquid: return "PPO Liquid";
            case AiAlgorithm::PpoImpala: return "PPO Impala";
            case AiAlgorithm::PpoLiquidImpala: return "PPO Liquid Impala";
        }
        return "PPO";
    }

    constexpr const char *shadow_label(const ShadowQuality quality) {
        switch (quality) {
            case ShadowQuality::Off: return "Shadows off";
            case ShadowQuality::Low: return "Low";
            case ShadowQuality::Medium: return "Medium";
            case ShadowQuality::High: return "High";
            case ShadowQuality::Ultra: return "Ultra";
        }
        return "Ultra";
    }

    inline std::string msaa_label(const int samples) {
        return samples <= 1 ? std::string("No AA") : std::to_string(samples) + "x";
    }

    constexpr const char *controller_label(const ControllerKind kind) {
        return kind == ControllerKind::Gamepad ? "Gamepad" : "Mouse + keyboard";
    }

    // mm:ss of a match duration
    inline std::string clock_label(const float seconds) {
        const int whole = std::max(0, static_cast<int>(seconds));
        char text[16];
        std::snprintf(text, sizeof(text), "%02d:%02d", whole / 60, whole % 60);
        return text;
    }

    inline std::string graphics_summary(const GameSettings &settings) {
        return std::string(settings.fullscreen ? "Fullscreen" : "Windowed") + " · "
               + msaa_label(settings.msaa_samples) + " · " + shadow_label(settings.shadow_quality);
    }

    inline std::string agent_summary(const GameSettings &settings, const bool agent_valid) {
        if (!agent_valid) return "No model loaded";
        return std::string(algorithm_label(settings.agent_algorithm)) + " · "
               + (settings.cuda ? "CUDA" : "CPU");
    }

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_LABELS_H
