#include "./agent_presets.h"

namespace arenai::desktop {

    std::vector<AgentPreset> agent_presets(const std::filesystem::path &resources_folder) {
        return {
            {.name = "PPO Liquid (debug model)",
             .algorithm = AiAlgorithm::PpoLiquid,
             .config = resources_folder / "trained_models" / "ppo_liquid" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo_liquid" / "save_18"},
            {.name = "PPO (debug model)",
             .algorithm = AiAlgorithm::Ppo,
             .config = resources_folder / "trained_models" / "ppo" / "config.json",
             .folder = resources_folder / "trained_models" / "ppo" / "save_3"},
            {.name = "PPG (debug model)",
             .algorithm = AiAlgorithm::Ppg,
             .config = resources_folder / "trained_models" / "ppg" / "config.json",
             .folder = resources_folder / "trained_models" / "ppg" / "save_1"},
        };
    }

}// namespace arenai::desktop
