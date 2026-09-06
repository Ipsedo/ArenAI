//
// Created by samuel on 22/01/2026.
//

#ifndef ARENAI_AGENT_HOST_FACTORY_H
#define ARENAI_AGENT_HOST_FACTORY_H

#include <format>
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
        template<typename T>
        T get_value(const std::string &argument_name, T default_value) {
            if (!agent_arguments.contains(argument_name)) return default_value;

            const auto &value = agent_arguments[argument_name];
            if (!value.is_string()) return value.get<T>();

            // runs dumped before the json migration stored every value as the
            // CLI string it came from
            const auto value_as_string = value.get<std::string>();
            std::stringstream ss(value_as_string);
            T parsed_value;
            ss >> parsed_value;

            if (ss.fail() || !ss.eof())
                throw std::runtime_error(std::format(
                    R"(Wrong value for "{}" : "{}", example : "{}")", argument_name,
                    value_as_string, default_value));

            return parsed_value;
        }

        template<typename T>
        T get_value(
            const std::string &argument_name, const std::function<T(std::string)> &parse_fn,
            T default_value) {
            if (!agent_arguments.contains(argument_name)) return default_value;

            return parse_fn(agent_arguments[argument_name].get<std::string>());
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
