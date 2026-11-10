#ifndef ARENAI_DESKTOP_CORE_SETTINGS_H
#define ARENAI_DESKTOP_CORE_SETTINGS_H

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "../controller/bindings.h"
#include "../controller/control_kind.h"

namespace arenai::desktop {

    // shadow quality presets; Ultra matches the historical hardcoded rendering
    enum class ShadowQuality { Off, Low, Medium, High, Ultra };

    // canonical names, shared by the JSON preferences and the menu bindings
    constexpr const char *to_string(const ShadowQuality quality) {
        switch (quality) {
            case ShadowQuality::Off: return "off";
            case ShadowQuality::Low: return "low";
            case ShadowQuality::Medium: return "medium";
            case ShadowQuality::High: return "high";
            default: return "ultra";
        }
    }

    constexpr std::optional<ShadowQuality> shadow_quality_from_string(std::string_view name) {
        for (const auto quality:
             {ShadowQuality::Off, ShadowQuality::Low, ShadowQuality::Medium, ShadowQuality::High,
              ShadowQuality::Ultra})
            if (name == to_string(quality)) return quality;
        return std::nullopt;
    }

    enum class AiAlgorithm { Ppo, PpoLiquid, Ppg };

    // canonical names, shared by the JSON preferences and the menu bindings
    constexpr const char *to_string(const AiAlgorithm algorithm) {
        switch (algorithm) {
            case AiAlgorithm::Ppo: return "ppo";
            case AiAlgorithm::Ppg: return "ppg";
            default: return "ppo_liquid";
        }
    }

    constexpr std::optional<AiAlgorithm> ai_algorithm_from_string(std::string_view name) {
        for (const auto algorithm: {AiAlgorithm::Ppo, AiAlgorithm::PpoLiquid, AiAlgorithm::Ppg})
            if (name == to_string(algorithm)) return algorithm;
        return std::nullopt;
    }

    // what the player can tune in the menu before launching a game
    struct GameSettings {
        int nb_tanks = 16;
        int spawn_side = 500;

        ControllerKind controller_kind = ControllerKind::Keyboard;
        ControlBindings bindings;

        bool fullscreen = false;

        ShadowQuality shadow_quality = ShadowQuality::Ultra;
        int msaa_samples = 4;

        std::string window_gpu;
        std::string vision_gpu;

        bool cuda = false;

        std::filesystem::path agent_folder;
        std::filesystem::path agent_config;
        AiAlgorithm agent_algorithm = AiAlgorithm::Ppo;
        std::optional<std::string> agent_preset;
    };

    struct AgentPreset {
        std::string name;
        AiAlgorithm algorithm;
        std::filesystem::path config;
        std::filesystem::path folder;
    };

    inline void apply_preset(const AgentPreset &preset, GameSettings &settings) {
        settings.agent_algorithm = preset.algorithm;
        settings.agent_config = preset.config;
        settings.agent_folder = preset.folder;
        settings.agent_preset = preset.name;
    }

    // the agent fields of GameSettings, as handed to the loader
    struct AgentSelection {
        std::filesystem::path config;// empty = auto-resolved config.json
        std::filesystem::path folder;
        AiAlgorithm algorithm;
    };

    inline AgentSelection agent_selection(const GameSettings &settings) {
        return {
            .config = settings.agent_config,
            .folder = settings.agent_folder,
            .algorithm = settings.agent_algorithm};
    }

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_SETTINGS_H
