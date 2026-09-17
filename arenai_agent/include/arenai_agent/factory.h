//
// Created by samuel on 22/01/2026.
//

#ifndef ARENAI_AGENT_HOST_FACTORY_H
#define ARENAI_AGENT_HOST_FACTORY_H

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "./agent.h"

namespace arenai::agent {

    enum AgentAlgorithm { SAC, PPO, PPO_LIQUID };

    class AgentFactory {
    public:
        // config: the content of a training run's config.json — the network
        // hyper-parameters come from its "agent" section, the vision size and
        // the control frequency from its "environment" section
        explicit AgentFactory(const nlohmann::json &config);

        std::shared_ptr<AbstractAgent> get_agent(
            AgentAlgorithm algorithm, const int &nb_sensors, const int &nb_continuous_actions,
            const int &nb_discrete_actions, bool cuda);

        int get_vision_height() const;
        int get_vision_width() const;
        float get_wanted_frequency() const;

    private:
        // every hyper-parameter is required: a config.json that misses one
        // does not describe the run it claims to (at() throws on a missing key)
        template<typename T>
        T get_value(const std::string &argument_name) {
            return agent_arguments.at(argument_name).get<T>();
        }

        std::shared_ptr<AbstractAgent> create_sac_agent(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
            bool cuda);

        std::shared_ptr<AbstractAgent> create_ppo_agent(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
            bool cuda);

        std::shared_ptr<AbstractAgent> create_liquid_ppo_agent(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
            bool cuda);

        nlohmann::json agent_arguments;
        int vision_height;
        int vision_width;
        float wanted_frequency;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_FACTORY_H
