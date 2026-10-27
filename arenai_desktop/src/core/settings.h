//
// Created by samuel on 24/10/2026.
//

#ifndef ARENAI_DESKTOP_CORE_SETTINGS_H
#define ARENAI_DESKTOP_CORE_SETTINGS_H

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "../controller/bindings.h"
#include "../controller/control_kind.h"

// The player-tunable configuration of a game, as the domain sees it: no UI
// type here, the gui/ hexagon reads and writes this struct through its port.
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

    // the RL algorithms the enemy agent can be built from; mirrors
    // agent::AgentAlgorithm without pulling arenai_agent into the settings
    enum class AiAlgorithm { Ppo, PpoLiquid, PpoImpala, PpoLiquidImpala };

    // canonical names, shared by the JSON preferences and the menu bindings
    constexpr const char *to_string(const AiAlgorithm algorithm) {
        switch (algorithm) {
            case AiAlgorithm::Ppo: return "ppo";
            case AiAlgorithm::PpoImpala: return "ppo_impala";
            case AiAlgorithm::PpoLiquidImpala: return "ppo_liquid_impala";
            default: return "ppo_liquid";
        }
    }

    constexpr std::optional<AiAlgorithm> ai_algorithm_from_string(std::string_view name) {
        for (const auto algorithm:
             {AiAlgorithm::Ppo, AiAlgorithm::PpoLiquid, AiAlgorithm::PpoImpala,
              AiAlgorithm::PpoLiquidImpala})
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

        // run the enemy agent's networks on CUDA; forced back to false when
        // no CUDA device is available on this machine
        bool cuda = false;

        // the trained enemy agent: its state-dict folder, the config.json of
        // its training run (empty = looked up next to the state dicts) and
        // the algorithm to rebuild the networks with
        std::filesystem::path agent_folder;
        std::filesystem::path agent_config;
        AiAlgorithm agent_algorithm = AiAlgorithm::Ppo;
        // the AgentPreset the three fields above were filled from, by name;
        // nullopt once the player edited any of them by hand ("Custom")
        std::optional<std::string> agent_preset;
    };

    // a ready-to-play bundle shipped with the game: picking one in the menu
    // fills the agent fields of GameSettings in one go
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
