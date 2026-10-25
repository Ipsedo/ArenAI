//
// Created by samuel on 19/07/2026.
//

#include "./agent_loading_checker.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <arenai_model/constants.h>

namespace arenai::desktop {

    namespace {

        nlohmann::json read_config(const AgentSelection &selection) {
            const auto config_path = resolve_agent_config(selection);
            if (!config_path)
                throw std::runtime_error(
                    "config.json not found in " + selection.folder.string() + " or its parent");

            std::ifstream config_stream(*config_path);
            if (!config_stream.is_open())
                throw std::runtime_error("cannot open " + config_path->string());
            return nlohmann::json::parse(config_stream);
        }

        std::shared_ptr<agent::AbstractAgent>
        build_agent(agent::AgentFactory &factory, const AiAlgorithm algorithm, const bool cuda) {
            return factory.get_agent(
                to_agent_algorithm(algorithm), model::ENEMY_PROPRIOCEPTION_SIZE,
                model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION, cuda);
        }

    }// namespace

    agent::AgentAlgorithm to_agent_algorithm(const AiAlgorithm algorithm) {
        switch (algorithm) {
            case AiAlgorithm::Ppo: return agent::PPO;
            default: return agent::PPO_LIQUID;
        }
    }

    std::optional<std::filesystem::path> resolve_agent_config(const AgentSelection &selection) {
        if (!selection.config.empty()) return selection.config;

        for (const auto &candidate:
             {selection.folder / "config.json", selection.folder.parent_path() / "config.json"})
            if (std::filesystem::is_regular_file(candidate)) return candidate;

        return std::nullopt;
    }

    std::optional<std::string> check_agent(const AgentSelection &selection) {
        try {
            agent::AgentFactory factory(read_config(selection));

            // stays on CPU: the point is only to prove the state dicts load
            // and the networks answer a forward pass
            const auto agent = build_agent(factory, selection.algorithm, false);
            agent->load(selection.folder);

            const core::State blank_state = {
                .vision =
                    {.pixels = std::vector<uint8_t>(
                         static_cast<size_t>(3) * factory.get_vision_height()
                             * factory.get_vision_width(),
                         0)},
                .proprioception = std::vector(model::ENEMY_PROPRIOCEPTION_SIZE, 0.f),
                .can_fire = true};
            agent->act({blank_state}, factory.get_vision_height(), factory.get_vision_width());

            return std::nullopt;
        } catch (const std::exception &e) { return e.what(); }
    }

    LoadedAgent load_agent(const AgentSelection &selection, const bool cuda) {
        agent::AgentFactory factory(read_config(selection));

        auto agent = build_agent(factory, selection.algorithm, cuda);
        agent->load(selection.folder);

        return {
            .agent = std::move(agent),
            .vision_height = factory.get_vision_height(),
            .vision_width = factory.get_vision_width(),
            .wanted_frequency = factory.get_wanted_frequency(),
            .action_repeat = factory.get_action_repeat()};
    }

}// namespace arenai::desktop
