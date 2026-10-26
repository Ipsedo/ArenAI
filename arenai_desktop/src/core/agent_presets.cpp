//
// Created by samuel on 24/10/2026.
//

#include "./agent_presets.h"

namespace arenai::desktop {

    std::vector<AgentPreset> agent_presets(const std::filesystem::path &resources_folder) {
        return {
            {.name = "PPO Liquid (debug model)",
             .algorithm = AiAlgorithm::PpoLiquid,
             .config = resources_folder / "trained_models" / "ppo_liquid" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo_liquid" / "save_7"},
            {.name = "PPO (debug model)",
             .algorithm = AiAlgorithm::Ppo,
             .config = resources_folder / "trained_models" / "ppo" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo" / "save_7"}};
    }

}// namespace arenai::desktop
