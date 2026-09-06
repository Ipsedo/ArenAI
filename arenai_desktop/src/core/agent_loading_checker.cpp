//
// Created by samuel on 19/07/2026.
//

#include "./agent_loading_checker.h"

#include <fstream>

#include <nlohmann/json.hpp>

#include <arenai_model/constants.h>

namespace arenai::desktop {

    agent::AgentAlgorithm to_agent_algorithm(const gui::AiAlgorithm algorithm) {
        switch (algorithm) {
            case gui::AiAlgorithm::Sac: return agent::SAC;
            case gui::AiAlgorithm::Ppo: return agent::PPO;
            default: return agent::PPO_LIQUID;
        }
    }

    std::optional<std::filesystem::path>
    resolve_agent_config(const gui::AgentSelection &selection) {
        if (!selection.config.empty()) return selection.config;

        for (const auto &candidate:
             {selection.folder / "config.json", selection.folder.parent_path() / "config.json"})
            if (std::filesystem::is_regular_file(candidate)) return candidate;

        return std::nullopt;
    }

    std::optional<std::string> check_agent(const gui::AgentSelection &selection) {
        const auto config_path = resolve_agent_config(selection);
        if (!config_path)
            return "config.json not found in " + selection.folder.string() + " or its parent";

        try {
            std::ifstream config_stream(*config_path);
            if (!config_stream.is_open())
                throw std::runtime_error("cannot open " + config_path->string());
            const auto config = nlohmann::json::parse(config_stream);

            agent::AgentFactory factory(config);

            // stays on CPU: the point is only to prove the state dicts load
            // and the networks answer a forward pass
            const auto agent = factory.get_agent(
                to_agent_algorithm(selection.algorithm), model::ENEMY_PROPRIOCEPTION_SIZE,
                model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION, false);

            agent->load(selection.folder);

            const core::State blank_state = {
                .vision =
                    {.pixels = std::vector<uint8_t>(
                         static_cast<size_t>(3) * factory.get_vision_height()
                             * factory.get_vision_width(),
                         0)},
                .proprioception = std::vector<float>(model::ENEMY_PROPRIOCEPTION_SIZE, 0.f)};
            agent->act({blank_state}, factory.get_vision_height(), factory.get_vision_width());

            return std::nullopt;
        } catch (const std::exception &e) { return e.what(); }
    }

}// namespace arenai::desktop
