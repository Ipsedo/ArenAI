//
// Created by samuel on 24/10/2026.
//

#include "./agent_presets.h"

namespace arenai::desktop {

    std::vector<AgentPreset> agent_presets(const std::filesystem::path &resources_folder) {
        return {
            {.name = "PPO Liquid (save 44)",
             .algorithm = AiAlgorithm::PpoLiquid,
             .config = resources_folder / "trained_models" / "ppo_liquid" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo_liquid" / "save_44"},
            {.name = "PPO (save 26)",
             .algorithm = AiAlgorithm::Ppo,
             .config = resources_folder / "trained_models" / "ppo" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo" / "save_26"}};
    }

}// namespace arenai::desktop
