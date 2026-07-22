//
// Created by samuel on 22/01/2026.
//

#ifndef ARENAI_AGENT_HOST_FACTORY_H
#define ARENAI_AGENT_HOST_FACTORY_H

#include <format>
#include <map>
#include <memory>
#include <string>

#include "./agent.h"

namespace arenai::agent {

    class AgentFactory {
    public:
        virtual ~AgentFactory() = default;

        explicit AgentFactory(const std::map<std::string, std::string> &arguments);

        std::shared_ptr<AbstractAgent> get_agent(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_actions);

    protected:
        template<typename T>
        T get_value(const std::string &argument_name, T default_value) {
            if (!arguments.contains(argument_name)) return default_value;

            const std::string value_as_string = arguments[argument_name];
            std::stringstream ss(value_as_string);
            T value;
            ss >> value;

            if (ss.fail() || !ss.eof())
                throw std::runtime_error(std::format(
                    R"(Wrong value for "{}" : "{}", example : "{}")", argument_name,
                    value_as_string, default_value));

            arguments.erase(arguments.find(argument_name));

            return value;
        }

        template<typename T>
        T get_value(
            const std::string &argument_name, const std::function<T(std::string)> &parse_fn,
            T default_value) {
            if (!arguments.contains(argument_name)) return default_value;

            const std::string value_as_string = arguments[argument_name];

            arguments.erase(arguments.find(argument_name));

            return parse_fn(value_as_string);
        }

        virtual std::shared_ptr<AbstractAgent> get_agent_impl(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_action) = 0;

    private:
        std::map<std::string, std::string> arguments;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_FACTORY_H
